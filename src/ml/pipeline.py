from logging import getLogger

import torch

from data.dataset import Dataset
from config import AppConfig
from ml.trainer import MLTrainer
from ml.predictor import MLPredictor
from sklearn.model_selection import PredefinedSplit
import ml.method as ml_method
from ml.summary_writer import MySummaryWriter
import tracemalloc
from ml.common import General, PerformanceMap, Results, Performance
from pathlib import Path
import numpy as np

logger = getLogger('app')


def log_results_performance(writer: MySummaryWriter, results: Results, cutoff: float, performance: Performance,
                            performance_map: PerformanceMap, results_path: Path, performance_path: Path):
    # log results and performance
    writer.add_protein_results(results, cutoff=cutoff)
    writer.add_performance_figures(performance)
    writer.add_performance_scalars(performance)

    cutoffs = np.linspace(0, 1, 10, endpoint=True)
    cutoffs = (cutoffs * 10).astype(int) / 10
    # writer.add_performance_per_cutoff(results, cutoffs=cutoffs)

    # save results into csv
    General.to_csv(df=results.to_df(cutoff=list(cutoffs)),
                   filename=results_path)
    General.to_csv(df=performance_map.to_df(),
                   filename=performance_path)


class Pipeline(object):

    @staticmethod
    def cross_training(config: AppConfig, dataset: Dataset, tag: str, max_length: int) -> Results:
        log_tracemalloc = 'tracemalloc' in config.get_log() and config.get_log()['tracemalloc']
        snapshot1 = None
        if log_tracemalloc:
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()

        # Read config
        ml_params = config.get_ml_params()
        method_params = config.get_method_params()

        # Prepare data
        logger.info("Prepare data")

        fold_array = dataset.fold_array
        prot_ids = dataset.prot_ids
        ps = PredefinedSplit(fold_array)

        # Prepare logging
        model_dir = config.get_ml_model_path() / tag
        predictions_dir = config.get_ml_predictions_path() / tag
        stats_dir = config.get_ml_stats_path() / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Start training
        logger.info("Start training")
        validation_results = Results()
        validation_performance_map = PerformanceMap()
        split_counter = 0
        for train_index, validation_index in ps.split():
            logger.info("Split: " + str(split_counter + 1))
            if split_counter == ml_params["num_cross_splits"]:
                break
            split_counter += 1
            # Prepare trainer
            logger.info("Prepare trainer")
            method = ml_method.MethodName.method(method_params=method_params, dataset=dataset, max_length=max_length)
            split_counter = fold_array[validation_index[0]]
            train_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'train' / f'split_{str(split_counter)}')
            validation_writer = MySummaryWriter(
                output_dir=stats_dir / 'train_set' / 'predict' / f'split_{str(split_counter)}')
            performance_file_path = model_dir / f'model_{str(split_counter)}_perf.csv'
            model_file_path = model_dir / f'model_{str(split_counter)}.pt'
            results_file_path = predictions_dir / f'validation_{str(split_counter)}.csv'

            trainer = MLTrainer(dataset=dataset, method=method, ml_params=ml_params, train_writer=train_writer,
                                val_writer=validation_writer, model_file_path=model_file_path,
                                performance_file_path=performance_file_path, results_file_path=results_file_path)

            # Prepare split run
            train_ids = [prot_ids[train_idx] for train_idx in train_index]
            validation_ids = [prot_ids[test_idx] for test_idx in validation_index]

            # train model
            logger.info("Train model")
            _, validation_results_i = trainer(train_ids=train_ids, validation_ids=validation_ids)
            validation_results.merge(validation_results_i)
            logger.info("Calculating model performance")
            validation_performance_map.append_performance(
                validation_results_i.get_performance(cutoff=ml_params['cutoff']),
                tag=f'model_{str(split_counter)}')

            if log_tracemalloc:
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                mem_stats = "[ Top 10 differences ]\n"
                for stat in top_stats[:10]:
                    mem_stats += str(stat) + "\n"
                logger.warning(mem_stats)

        logger.info("Calculating total performance")
        total_validation_performance = validation_results.get_performance(cutoff=ml_params['cutoff'])
        validation_performance_map.append_performance(total_validation_performance, tag=f'model_total')

        logger.info("Logging final results and performance")
        # log results and performance
        total_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'predict' / 'total')
        log_results_performance(writer=total_writer, results=validation_results,
                                performance_map=validation_performance_map, performance=total_validation_performance,
                                cutoff=ml_params['cutoff'], results_path=predictions_dir / f'validation_total.csv',
                                performance_path=model_dir / f'model_validation_perf.csv')
        return validation_results

    @staticmethod
    def testing(config: AppConfig, dataset: Dataset, tag: str, max_length: int) -> Results:
        logger.info("Prepare data")
        prot_ids = dataset.prot_ids

        ml_params = config.get_ml_params()
        method_params = config.get_method_params()

        model_dir = config.get_ml_model_path() / tag
        stats_dir = config.get_ml_stats_path() / tag
        predictions_dir = config.get_ml_predictions_path() / tag

        results = Results()
        performance_map = PerformanceMap()
        for split_counter in range(1, ml_params["num_cross_splits"] + 1):  # test all 5 models
            logger.info("Split: " + str((split_counter + 1)))
            # load model
            model = torch.load(model_dir / f'model_{str(split_counter)}.pt')
            method = ml_method.MethodName.method(method_params=method_params, dataset=dataset, max_length=max_length)
            method.model = model
            predict_writer = MySummaryWriter(
                output_dir=stats_dir / 'test_set' / 'predict' / f'model_{str(split_counter)}')
            predictor = MLPredictor(dataset=dataset, method=method, writer=predict_writer, params=ml_params,
                                    tag=f'test')

            logger.info("Calculate predictions per protein")
            results_i = predictor(ids=prot_ids)
            logger.info("Calculating model performance")
            General.to_csv(df=results_i.to_df(cutoff=ml_params['cutoff']),
                           filename=predictions_dir / f'test_model_{str(split_counter)}.csv')
            performance_map.append_performance(results_i.get_performance(cutoff=ml_params['cutoff']),
                                               tag=f'model_{str(split_counter)}')
            for k in results_i.keys():
                if k in results.keys():
                    prot_result = results[k]
                    prot_result.add(results_i[k])
                else:
                    results[k] = results_i[k]

        for k in results.keys():
            results[k].normalize(ml_params["num_cross_splits"])

        logger.info("Calculating total performance")
        total_performance = results.get_performance(cutoff=ml_params['cutoff'])
        performance_map.append_performance(total_performance, tag=f'model_total')

        logger.info("Logging final results and performance")
        # log results and performance
        total_writer = MySummaryWriter(output_dir=stats_dir / 'test_set' / 'predict' / 'normalized')
        log_results_performance(writer=total_writer, results=results, performance_map=performance_map,
                                performance=total_performance, cutoff=ml_params['cutoff'],
                                results_path=predictions_dir / f'test_total.csv',
                                performance_path=model_dir / f'model_test_perf.csv')
        return results

    @staticmethod
    def hyperparameter_optimization():
        """
        TODO
        :return:
        """
        pass
