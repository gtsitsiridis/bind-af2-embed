from logging import getLogger

import pandas as pd
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
from ml.template import RunTemplate

logger = getLogger('app')


class Pipeline(object):

    def __init__(self, config: AppConfig, template: RunTemplate, train_dataset: Dataset, test_dataset: Dataset,
                 max_length: int, tag: str):
        self._template = template
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._max_length = max_length
        self._log_tracemalloc = config.log.tracemalloc

        self._num_splits = config.input.params.get("num_cross_splits", 5)

        # Prepare logging paths
        self._model_dir = config.output.model_path / tag
        self._predictions_dir = config.output.predictions_path / tag
        self._stats_dir = config.output.stats_path / tag
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._predictions_dir.mkdir(parents=True, exist_ok=True)

    def cross_training(self) -> Results:
        cutoff = self._template.train_params.get('cutoff', 0.5)
        dataset = self._train_dataset
        model_dir = self._model_dir
        stats_dir = self._stats_dir
        predictions_dir = self._predictions_dir
        template = self._template
        max_length = self._max_length
        log_tracemalloc = self._log_tracemalloc
        num_splits = self._num_splits

        # Prepare data
        logger.info("Prepare data")
        fold_array = dataset.fold_array
        prot_ids = dataset.prot_ids
        ps = PredefinedSplit(fold_array)

        # Start training
        logger.info("Start training")
        validation_results = Results()
        validation_performance_map = PerformanceMap()
        split_counter = 0
        for train_index, validation_index in ps.split():
            if split_counter == num_splits:
                break
            logger.info("Split: " + str(split_counter + 1))

            split_counter += 1
            # Prepare trainer
            logger.info("Prepare trainer")
            method = ml_method.Method.get_method(template=template, dataset=dataset, max_length=max_length)
            split_counter = fold_array[validation_index[0]]
            train_writer = MySummaryWriter(
                output_dir=stats_dir / 'train_set' / 'train' / f'split_{str(split_counter)}')
            validation_writer = MySummaryWriter(
                output_dir=stats_dir / 'train_set' / 'predict' / f'split_{str(split_counter)}')
            performance_file_path = model_dir / f'model_{str(split_counter)}_perf.csv'
            model_file_path = model_dir / f'model_{str(split_counter)}.pt'
            results_file_path = predictions_dir / f'validation_{str(split_counter)}.csv'

            trainer = MLTrainer(dataset=dataset, method=method, template=template, train_writer=train_writer,
                                val_writer=validation_writer, model_file_path=model_file_path,
                                performance_file_path=performance_file_path, results_file_path=results_file_path,
                                log_tracemalloc=self._log_tracemalloc)

            # Prepare split run
            train_ids = [prot_ids[train_idx] for train_idx in train_index]
            validation_ids = [prot_ids[test_idx] for test_idx in validation_index]

            # train model
            logger.info("Train model")
            _, validation_results_i = trainer(train_ids=train_ids, validation_ids=validation_ids)
            validation_results.merge(validation_results_i)
            logger.info("Calculating model performance")
            validation_performance_map.append_performance(
                validation_results_i.get_performance(cutoff=cutoff),
                tag=f'model_{str(split_counter)}')

        logger.info("Calculating total performance")
        total_validation_performance = validation_results.get_performance(cutoff=cutoff)
        validation_performance_map.append_performance(total_validation_performance, tag=f'model_total')

        logger.info("Logging final results and performance")
        # log results and performance
        total_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'predict' / 'total')
        self.log_results_performance(writer=total_writer, results=validation_results,
                                     performance_map=validation_performance_map,
                                     performance=total_validation_performance,
                                     cutoff=cutoff, results_path=predictions_dir / f'validation_total.csv',
                                     performance_path=model_dir / f'model_validation_perf.csv')
        return validation_results

    def testing(self) -> Results:
        logger.info("Prepare data")
        dataset = self._test_dataset
        model_dir = self._model_dir
        stats_dir = self._stats_dir
        predictions_dir = self._predictions_dir
        num_splits = self._num_splits
        template = self._template
        max_length = self._max_length
        cutoff = self._template.train_params.get('cutoff', 0.5)

        prot_ids = dataset.prot_ids

        results = Results()
        performance_map = PerformanceMap()
        for split_counter in range(1, num_splits + 1):  # test all 5 models
            logger.info("Split: " + str(split_counter))
            # load model
            model = torch.load(model_dir / f'model_{str(split_counter)}.pt')
            method = ml_method.Method.get_method(template=template, dataset=dataset, max_length=max_length)
            method.model = model
            predict_writer = MySummaryWriter(
                output_dir=stats_dir / 'test_set' / 'predict' / f'model_{str(split_counter)}')
            predictor = MLPredictor(dataset=dataset, method=method, writer=predict_writer,
                                    template=template, tag=f'test')

            logger.info("Calculate predictions per protein")
            results_i = predictor(ids=prot_ids)
            logger.info("Calculating model performance")
            General.to_csv(df=results_i.to_df(cutoff=cutoff),
                           filename=predictions_dir / f'test_model_{str(split_counter)}.csv')
            performance_map.append_performance(results_i.get_performance(cutoff=cutoff),
                                               tag=f'model_{str(split_counter)}')
            for k in results_i.keys():
                if k in results.keys():
                    prot_result = results[k]
                    prot_result.add(results_i[k])
                else:
                    results[k] = results_i[k]

        for k in results.keys():
            results[k].normalize(num_splits)

        logger.info("Calculating total performance")
        total_performance = results.get_performance(cutoff=cutoff)
        performance_map.append_performance(total_performance, tag=f'model_total')

        logger.info("Logging final results and performance")
        # log results and performance
        total_writer = MySummaryWriter(output_dir=stats_dir / 'test_set' / 'predict' / 'normalized')
        self.log_results_performance(writer=total_writer, results=results, performance_map=performance_map,
                                     performance=total_performance, cutoff=cutoff,
                                     results_path=predictions_dir / f'test_total.csv',
                                     performance_path=model_dir / f'model_test_perf.csv')
        return results

    def log_results_performance(self, writer: MySummaryWriter, results: Results, cutoff: float,
                                performance: Performance,
                                performance_map: PerformanceMap, results_path: Path, performance_path: Path):
        # log results and performance
        writer.add_protein_results(results, cutoff=cutoff)
        writer.add_performance_figures(performance)
        writer.add_performance_scalars(performance)
        self.log_hparams(performance=performance, writer=writer)

        cutoffs = np.linspace(0, 1, 10, endpoint=True)
        cutoffs = (cutoffs * 10).astype(int) / 10

        # save results into csv
        General.to_csv(df=results.to_df(cutoff=list(cutoffs)),
                       filename=results_path)
        General.to_csv(df=performance_map.to_df(),
                       filename=performance_path)

    def log_hparams(self, performance: Performance, writer: MySummaryWriter):
        model_dir = self._model_dir
        template = self._template
        train_dataset_stats = {"train_" + k: v for k, v in self._train_dataset.summary().items()}
        test_dataset_stats = {"test_" + k: v for k, v in self._test_dataset.summary().items()}

        hparams = {"num_splits": self._num_splits}
        hparams.update(template.optimizer_params)
        hparams.update(template.model_params)
        hparams.update(template.train_params)
        hparams.update(train_dataset_stats)
        hparams.update(test_dataset_stats)

        # flatten loss params
        loss_params = template.loss_params.copy()
        if 'weights' in loss_params:
            loss_params['weights'] = ';'.join([str(x) for x in loss_params['weights']])
        if 'pos_weights' in loss_params:
            loss_params['pos_weights'] = ';'.join([str(x) for x in loss_params['pos_weights']])
        hparams.update(loss_params)

        # log to tensorboard
        writer.add_hparams(hparams=hparams, performance=performance)

        # log to file
        filename = model_dir / 'hparams.csv'
        General.to_csv(df=pd.DataFrame(hparams, index=[0]), filename=filename)
