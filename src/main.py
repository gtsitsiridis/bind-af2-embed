from config import AppConfig, InputConfig
from utils import Logging
from logging import getLogger
from ml.pipeline import Pipeline
from datetime import datetime
from data.dataset import Dataset
import argparse
from ml.template import RunTemplate

logger = getLogger('app')


def train(tag: str, config: AppConfig, template: RunTemplate, dataset: Dataset):
    logger.info(f'Tag {tag}')
    logger.info(str(template))

    pipeline = Pipeline(tag=tag, config=config, template=template, dataset=dataset)

    logger.info("Starting training")
    pipeline.cross_training()
    logger.info(f"Tag {tag}: Training done")


def run_test(tag: str, config: AppConfig, template: RunTemplate, dataset: Dataset):
    logger.info(f'Tag {tag}')
    logger.info(str(template))

    pipeline = Pipeline(tag=tag, config=config, template=template, dataset=dataset)

    logger.info("Starting testing")
    pipeline.testing()
    logger.info(f"Tag {tag}:Evaluation done")


def init_app(config_file: str, log: bool) -> AppConfig:
    config = AppConfig(config_file=config_file)
    Logging.setup_app_logger(config=config.log, write=log)
    logger.info("Input params: " + str(config.input.params))

    return config


def init_dataset(config: InputConfig, test: bool = False) -> Dataset:
    full_dataset = Dataset.full_dataset(config)
    logger.info("Total dataset:" + str(full_dataset.summary()))
    if test:
        dataset = full_dataset.get_subset(mode='test', config=config)
    else:
        dataset = full_dataset.get_subset(mode='train', config=config)

    logger.info("Dataset:" + str(dataset.summary()))
    return dataset


def __main():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--config', required=False)
    parser.add_argument('--template', required=True)
    parser.add_argument('--tag', required=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
    args = parser.parse_args()

    config = init_app(config_file=args.config, log=args.log)

    def _run(_template):
        tag = args.tag
        if args.test:
            assert tag is not None, "You need to provide the tag of a trained model to test."
            dataset = init_dataset(config=config.input, test=True)
            run_test(tag=tag, config=config, template=_template, dataset=dataset)
        else:
            if tag is None:
                tag = f'{datetime.now().strftime("%Y%m%d%H%M")}_{_template.name}'
            dataset = init_dataset(config=config.input)
            train(tag=tag, config=config, template=_template, dataset=dataset)

    if args.template == 'all':
        for template in config.iter_templates():
            _run(template)
    else:
        template = config.resolve_template(args.template)
        _run(template)


if __name__ == '__main__':
    __main()
