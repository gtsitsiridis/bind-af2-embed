import torchsummary

from config import AppConfig
from utils import Logging
from logging import getLogger
from ml.pipeline import Pipeline
from datetime import datetime
from data.dataset import Dataset
import argparse
import os

logger = getLogger('app')


def ml_pipeline(method_config: str, config_dir: str = None, log: bool = True):
    config = AppConfig(method_config, config_dir=config_dir)
    ml_params = config.get_ml_params()
    method_params = config.get_method_params()
    Logging.setup_app_logger(config=config, write=log)
    tag = f'{datetime.now().strftime("%Y%m%d%H%M")}_{method_config}'
    logger.info(f'Tag {tag}')
    logger.info(ml_params)
    logger.info(method_params)

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


#
# def test_cnn1d_model():
#     model = models.CNN1DModel(in_channels=1024, feature_channels=128, kernel_size=5, dropout=0.7)
#     torchsummary.summary(model, input_size=(1024, 500))
#
#
# def test_cnn2d_model():
#     model = models.DistMapsModel(500)
#     torchsummary.summary(model, input_size=(2, 500, 500))
#
#
# def test_combined_model():
#     model = models.CombinedModel(500)
#     torchsummary.summary(model, input_size=[(1024, 500), (2, 500, 500)])


def __main():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--log', action='store_true', default=False)
    args = parser.parse_args()

    if args.method == 'all':
        methods = [method.split('.')[0] for method in os.listdir(args.config + '/methods')]
        for method in methods:
            ml_pipeline(config_dir=args.config, log=args.log, method_config=method)
    else:
        ml_pipeline(config_dir=args.config, log=args.log, method_config=args.method)


if __name__ == '__main__':
    __main()
