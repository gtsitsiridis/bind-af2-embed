import logging
from config import AppConfig
from pathlib import Path
from logging import getLogger

logger = getLogger('app')


class FileUtils(object):
    @staticmethod
    def read_ids(ids_file) -> list:
        ids = []
        with open(ids_file) as read_in:
            for line in read_in:
                ids.append(line.strip())
        return ids

    @staticmethod
    def read_split_ids(splits_ids_files: list) -> (list, list):
        """

        :param splits_ids_files:
        :param subset:
        :return:
        """
        ids = []
        fold_array = []
        for ids_file in splits_ids_files:
            s = ids_file['id']
            file = ids_file['file']
            split_ids = FileUtils.read_ids(file)
            ids += split_ids
            fold_array += [s] * len(split_ids)
        if len(ids) != len(fold_array):
            print('WHAAAAT?')
        return ids, fold_array

    @staticmethod
    def read_binding_residues(file_in):
        """
        Read binding residues from file
        :param file_in:
        :return:
        """
        binding = dict()

        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                if len(splitted_line) > 1:
                    identifier = splitted_line[0]
                    residues = splitted_line[1].split(',')
                    residues_int = [int(r) for r in residues]

                    binding[identifier] = residues_int

        return binding


class Logging(object):
    @staticmethod
    def setup_app_logger(config: AppConfig, write: bool = False):
        log_config = config.get_log()
        assert log_config['loggers']['app']['level'], 'level has not been defined in the app logger configuration'

        level = log_config['loggers']['app']['level']

        logger = logging.getLogger('app')
        logger.setLevel(level)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

        if write:
            assert log_config['path'], 'path has not been defined in the log configuration'
            file_path = Path(log_config['path']) / 'app.log'
            fileHandler = logging.FileHandler(file_path)
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)
