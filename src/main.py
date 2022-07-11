from config import AppConfig, InputConfig
from utils import Logging
from logging import getLogger
from ml.pipeline import Pipeline
from datetime import datetime
from data.dataset import Dataset
import argparse
from ml.template import RunTemplate

logger = getLogger('app')


def train_test(config: AppConfig, template: RunTemplate, train_dataset: Dataset, test_dataset: Dataset,
               max_length: int):
    tag = f'{datetime.now().strftime("%Y%m%d%H%M")}_{template.name}'
    logger.info(f'Tag {tag}')
    logger.info(str(template))

    pipeline = Pipeline(tag=tag, config=config, template=template, max_length=max_length, train_dataset=train_dataset,
                        test_dataset=test_dataset)

    logger.info("Starting training")
    pipeline.cross_training()
    logger.info("Training done")

    logger.info("Starting testing")
    pipeline.testing()
    logger.info("Evaluation done")


def init_app(config_file: str, log: bool) -> AppConfig:
    config = AppConfig(config_file=config_file)
    Logging.setup_app_logger(config=config.log, write=log)
    logger.info("Input params: " + str(config.input.params))

    return config


def init_dataset(config: InputConfig) -> (Dataset, Dataset, int):
    full_dataset = Dataset.full_dataset(config)
    logger.info("Total dataset:" + str(full_dataset.summary()))
    test_dataset = full_dataset.get_subset(mode='test', config=config)
    train_dataset = full_dataset.get_subset(mode='train', config=config)

    logger.info("Test dataset:" + str(test_dataset.summary()))
    logger.info("Train dataset:" + str(train_dataset.summary()))
    logger.info("Filtered proteins: " + str(len(full_dataset) - len(train_dataset) - len(test_dataset)))
    max_length = max(test_dataset.determine_max_length(), train_dataset.determine_max_length())
    del full_dataset

    return train_dataset, test_dataset, max_length


def __main():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--config', required=False)
    parser.add_argument('--template', required=True)
    parser.add_argument('--log', action='store_true', default=False)
    args = parser.parse_args()

    config = init_app(config_file=args.config, log=args.log)
    train_dataset, test_dataset, max_length = init_dataset(config=config.input)

    if args.template == 'all':
        for template in config.iter_templates():
            train_test(config=config, template=template, train_dataset=train_dataset, test_dataset=test_dataset,
                       max_length=max_length)
    else:
        template = config.resolve_template(args.template)
        train_test(config=config, template=template, train_dataset=train_dataset, test_dataset=test_dataset,
                   max_length=max_length)


if __name__ == '__main__':
    __main()
