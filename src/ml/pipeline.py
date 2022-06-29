from logging import getLogger

import torch

from data.dataset import Dataset
from config import AppConfig
from ml.trainer import MLTrainer
from ml.predictor import MLPredictor
from sklearn.model_selection import PredefinedSplit
from ml.method import Method, CNN1DMethod, MethodName
from ml.summary_writer import MySummaryWriter
from ml.predictor import Results

logger = getLogger('app')


class Pipeline(object):
    @staticmethod
    def cross_training(config: AppConfig, method_name: MethodName, tag: str, subset: int = -1) -> Results:
        # Prepare data
        logger.info("Prepare data")
        dataset = Dataset(config, mode='train', subset=subset)
        fold_array = dataset.fold_array
        prot_ids = dataset.prot_ids
        ps = PredefinedSplit(fold_array)
        ml_config = config.get_ml()

        # Prepare trainer
        logger.info("Prepare trainer")
        method = Pipeline.name_to_method(name=method_name, ml_config=ml_config,
                                         max_length=dataset.determine_max_length(),
                                         embedding_size=dataset.embedding_size)
        trainer = MLTrainer(dataset=dataset, method=method, train_params=config.get_ml_train_params())
        predictor = MLPredictor(dataset=dataset, method=method)

        # Prepare logging
        model_dir = config.get_ml_model_path() / tag
        stats_dir = config.get_ml_stats_path() / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Start training
        logger.info("Start training")
        results = Results()
        for train_index, validation_index in ps.split():
            method.reset()
            # Prepare split run
            split_counter = fold_array[validation_index[0]]
            train_ids = [prot_ids[train_idx] for train_idx in train_index]
            validation_ids = [prot_ids[test_idx] for test_idx in validation_index]
            train_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'train' / f'split_{str(split_counter)}')

            logger.info("Train model")
            train_validate_performance = trainer.run(train_ids=train_ids, validation_ids=validation_ids,
                                                                  writer=train_writer)

            # save model
            torch.save(method.model, model_dir / f'checkpoint_{str(split_counter)}.pt')

            logger.info("Calculate predictions per protein")
            curr_results = predictor.run(ids=validation_ids)
            predict_writer = MySummaryWriter(
                output_dir=stats_dir / 'train_set' / 'predict' / f'split_{str(split_counter)}')
            predict_writer.add_protein_results(curr_results)

            results.merge(curr_results)

        total_predict_writer = MySummaryWriter(output_dir=stats_dir / 'train_set' / 'predict' / 'total')
        total_predict_writer.add_protein_results(results)
        return results

    @staticmethod
    def testing(config: AppConfig, method_name: MethodName, tag: str) -> Results:
        logger.info("Prepare data")
        dataset = Dataset(config, mode='test')
        prot_ids = dataset.prot_ids

        method = Pipeline.name_to_method(name=method_name, ml_config=config.get_ml(),
                                         max_length=dataset.determine_max_length(),
                                         embedding_size=dataset.embedding_size)

        model_dir = config.get_ml_model_path() / tag
        stats_dir = config.get_ml_stats_path() / tag

        predictor = MLPredictor(dataset=dataset, method=method)
        results = Results()
        for split_counter in range(1, 6):  # test all 5 models
            # load model
            model = torch.load(model_dir / f'checkpoint_{str(split_counter)}.pt')

            logger.info("Calculate predictions per protein")
            curr_results = predictor.run(ids=prot_ids, model=model)
            predict_writer = MySummaryWriter(
                output_dir=stats_dir / 'test_set' / 'predict' / f'model_{str(split_counter)}')
            predict_writer.add_protein_results(curr_results)

            for k in curr_results.keys():
                if k in results.keys():
                    prot_result = results[k]
                    prot_result.add_predictions(curr_results[k].predictions)
                else:
                    results[k] = curr_results[k]

        for k in results.keys():
            results[k].normalize_predictions(5)

        total_predict_writer = MySummaryWriter(output_dir=stats_dir / 'test_set' / 'predict' / 'normalized')
        total_predict_writer.add_protein_results(results)
        return results

    @staticmethod
    def hyperparameter_optimization():
        """
        TODO
        :return:
        """
        pass

    @staticmethod
    def name_to_method(name: MethodName, max_length: int, embedding_size: int, ml_config: dict) -> Method:
        if name == MethodName.CNN1D:
            return CNN1DMethod(max_length=max_length, embedding_size=embedding_size, ml_config=ml_config)
