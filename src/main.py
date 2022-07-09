import torchsummary

from config import AppConfig
from utils import Logging
from logging import getLogger
from ml.pipeline import Pipeline
from ml.method import MethodName
from datetime import datetime
import ml.models as models
from data.dataset import Dataset
import argparse

logger = getLogger('app')


def ml_pipeline(config_file: str = None, log: bool = True):
    config = AppConfig(config_file=config_file)
    ml_params = config.get_ml_params()
    method_params = config.get_method_params()
    Logging.setup_app_logger(config=config, write=log)
    method_name = MethodName[method_params['name']]
    tag = f'{datetime.now().strftime("%Y%m%d%H%M")}_{method_name.name}'
    logger.info(f'Tag {tag}')
    logger.info(ml_params)

    dataset = Dataset.dataset_from_config(config, mode='all')
    logger.info("Total dataset:" + dataset.summary())
    test_dataset = dataset.get_subset(mode='test', config=config, plddt_limit=ml_params['plddt_limit'],
                                      max_length=ml_params['max_length'], min_length=ml_params['min_length'])
    train_dataset = dataset.get_subset(mode='train', config=config, subset=ml_params['subset'],
                                       plddt_limit=ml_params['plddt_limit'], max_length=ml_params['max_length'],
                                       min_length=ml_params['min_length'])
    logger.info("Test dataset:" + test_dataset.summary())
    logger.info("Train dataset:" + train_dataset.summary())
    logger.info("Filtered proteins: " + str(len(dataset) - (len(train_dataset) + len(test_dataset))))

    max_length = max(test_dataset.determine_max_length(), train_dataset.determine_max_length())
    del dataset

    Pipeline.cross_training(config=config, tag=tag, dataset=train_dataset, max_length=max_length)
    logger.info("Training done")

    Pipeline.testing(config=config, tag=tag, dataset=test_dataset, max_length=max_length)
    logger.info("Evaluation done")


def test_cnn1d_model():
    model = models.CNN1DModel(in_channels=1024, feature_channels=128, kernel_size=5, dropout=0.7)
    torchsummary.summary(model, input_size=(1024, 500))


def test_cnn2d_model():
    model = models.CNN2DModel(500)
    torchsummary.summary(model, input_size=(2, 500, 500))


def test_combined_model():
    model = models.CNNCombinedModel(500)
    torchsummary.summary(model, input_size=[(1024, 500), (2, 500, 500)])


def __main():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--config', required=False)
    parser.add_argument('--log', action='store_true', default=False)
    args = parser.parse_args()
    ml_pipeline(config_file=args.config, log=args.log)


if __name__ == '__main__':
    __main()
