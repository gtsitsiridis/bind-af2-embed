from logging import getLogger

import torch

from data.dataset import Dataset
from config import AppConfig
from ml.trainer import MLTrainer
from ml.predictor import MLPredictor
from sklearn.model_selection import PredefinedSplit
from ml.method import Method, CNN1DAllMethod, CNN1DEmbeddingsMethod, MethodName
from ml.summary_writer import MySummaryWriter
from ml.predictor import Results
import tracemalloc
from ml.common import General

logger = getLogger('app')


class Pipeline(object):
    @staticmethod
    def cross_training(config: AppConfig, method_name: MethodName, tag: str) -> Results:
        log_tracemalloc = 'tracemalloc' in config.get_log() and config.get_log()['tracemalloc']
        snapshot1 = None
        if log_tracemalloc:
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()

        # Read config
        params = config.get_ml_params()
        ml_config = config.get_ml()

        # Prepare data
        logger.info("Prepare data")
        dataset = Dataset(config, mode='train', subset=params['subset'])
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
        results = Results()
        for train_index, validation_index in ps.split():
            # Prepare trainer
            logger.info("Prepare trainer")
            method = Pipeline.name_to_method(name=method_name, ml_config=ml_config, dataset=dataset)
            split_counter = fold_array[validation_index[0]]
            train_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'train' / f'split_{str(split_counter)}')
            validation_writer = MySummaryWriter(
                output_dir=stats_dir / 'train_set' / 'predict' / f'split_{str(split_counter)}')
            performance_file_path = model_dir / f'model_{str(split_counter)}_perf.csv'
            model_file_path = model_dir / f'model_{str(split_counter)}.pt'

            trainer = MLTrainer(dataset=dataset, method=method, params=params, train_writer=train_writer,
                                val_writer=validation_writer, model_file_path=model_file_path,
                                performance_file_path=performance_file_path)

            # Prepare split run
            train_ids = [prot_ids[train_idx] for train_idx in train_index]
            validation_ids = [prot_ids[test_idx] for test_idx in validation_index]

            # train model
            logger.info("Train model")
            _, validation_results = trainer(train_ids=train_ids, validation_ids=validation_ids)
            results.merge(validation_results)

            if log_tracemalloc:
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                mem_stats = "[ Top 10 differences ]\n"
                for stat in top_stats[:10]:
                    mem_stats += str(stat) + "\n"
                logger.warning(mem_stats)

        # log results and performance
        total_predict_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'predict' / 'total')
        total_predict_writer.add_protein_results(results, cutoff=params['cutoff'])
        total_predict_writer.add_single_performance(results.get_single_performance(cutoff=params['cutoff']))
        # save results into csv
        General.to_csv(df=results.to_df(cutoff=params['cutoff']), filename=predictions_dir / f'train_total.csv')
        return results

    @staticmethod
    def testing(config: AppConfig, method_name: MethodName, tag: str) -> Results:
        logger.info("Prepare data")
        dataset = Dataset(config, mode='test')
        prot_ids = dataset.prot_ids

        params = config.get_ml_params()
        model_dir = config.get_ml_model_path() / tag
        stats_dir = config.get_ml_stats_path() / tag
        predictions_dir = config.get_ml_predictions_path() / tag

        results = Results()
        for split_counter in range(1, 6):  # test all 5 models
            # load model
            model = torch.load(model_dir / f'model_{str(split_counter)}.pt')
            method = Pipeline.name_to_method(name=method_name, ml_config=config.get_ml(),
                                             dataset=dataset)
            method.model = model
            predict_writer = MySummaryWriter(
                output_dir=stats_dir / 'test_set' / 'predict' / f'model_{str(split_counter)}')
            predictor = MLPredictor(dataset=dataset, method=method, writer=predict_writer, params=params,
                                    tag=f'test')

            logger.info("Calculate predictions per protein")
            curr_results = predictor(ids=prot_ids)
            General.to_csv(df=
                           curr_results.to_df(cutoff=params['cutoff']),
                           filename=predictions_dir / f'test_model_{str(split_counter)}.csv')

            predict_writer.add_protein_results(curr_results, cutoff=params['cutoff'])
            predict_writer.add_single_performance(curr_results.get_single_performance(cutoff=params['cutoff']))

            for k in curr_results.keys():
                if k in results.keys():
                    prot_result = results[k]
                    prot_result.add(curr_results[k])
                else:
                    results[k] = curr_results[k]

        for k in results.keys():
            results[k].normalize(5)

        total_predict_writer = MySummaryWriter(output_dir=stats_dir / 'test_set' / 'predict' / 'normalized')
        total_predict_writer.add_protein_results(results, cutoff=params['cutoff'])
        total_predict_writer.add_single_performance(results.get_single_performance(cutoff=params['cutoff']))
        General.to_csv(df=results.to_df(cutoff=params['cutoff']),
                       filename=predictions_dir / f'test_norm.csv')
        return results

    @staticmethod
    def hyperparameter_optimization():
        """
        TODO
        :return:
        """
        pass

    @staticmethod
    def name_to_method(name: MethodName, ml_config: dict, dataset: Dataset) -> Method:
        if name == MethodName.CNN1D_ALL:
            return CNN1DAllMethod(dataset=dataset, ml_config=ml_config)
        elif name == MethodName.CNN1D_EMBEDDINGS:
            return CNN1DEmbeddingsMethod(dataset=dataset, ml_config=ml_config)
        else:
            assert False, f"The method {name.name} has not been defined yet."
