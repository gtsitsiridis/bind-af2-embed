from logging import getLogger
from data.dataset import Dataset
from config import AppConfig
from trainer import MLTrainer
from sklearn.model_selection import PredefinedSplit
from datetime import date
from pathlib import Path
from method import Method

logger = getLogger('app')


class Pipeline(object):
    @staticmethod
    def cross_training(config: AppConfig, method: Method):
        logger.info("Prepare data")
        dataset = Dataset(config, mode='train')
        fold_array = dataset.fold_array
        prot_ids = dataset.prot_ids

        ps = PredefinedSplit(fold_array)
        ml_config = config.get_ml()
        trainer = MLTrainer(dataset=dataset)
        output_dir = Pipeline._get_output_dir(root_output_dir=ml_config['output_dir'], method_name=method.name)
        proteins = dict()
        for train_index, validation_index in ps.split():
            split_counter = fold_array[validation_index[0]]
            split_output_dir = output_dir / f'{split_counter}/'
            split_output_dir.mkdir()
            train_ids = [prot_ids[train_idx] for train_idx in train_index]
            validation_ids = [prot_ids[test_idx] for test_idx in validation_index]

            logger.info("Train model")
            model_split = trainer.train(train_ids=train_ids, validation_ids=validation_ids,
                                        verbose=False, output_dir=split_output_dir)
            logger.info("Calculate predictions per protein")
            curr_proteins = trainer.predict(ids=validation_ids, model=model_split)
            proteins = {**proteins, **curr_proteins}

            # print("Calculate predictions per protein")
            # ml_predictor = MLPredictor(model_split)
            # curr_proteins = ml_predictor.predict_per_protein(validation_ids, sequences, embeddings, labels, max_length)
            #
            # proteins = {**proteins, **curr_proteins}
            #
            # if predictions_output is not None:
            #     FileManager.write_predictions(proteins, predictions_output, 0.5, ri)

    @staticmethod
    def _get_output_dir(root_output_dir: str, method_name: str) -> Path:
        today = date.today()
        new_dir = today.strftime("%Y%m%d%H%M")
        new_path = (Path(root_output_dir) / method_name / new_dir)
        new_path.mkdir(parents=True, exist_ok=True)
        return new_path

    @staticmethod
    def hyperparameter_optimization():
        """
        TODO
        :return:
        """
        pass

    @staticmethod
    def prediction():
        """
        TODO
        :return:
        """
        pass
