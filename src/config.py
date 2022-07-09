import json
from pathlib import Path


class AppConfig(object):
    def __init__(self, config_file: str = None):
        if config_file is None:
            config_file = '../config.json'
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get_input(self) -> dict:
        return self.config['input']

    def get_log(self) -> dict:
        return self.config['log']

    def get_ml(self) -> dict:
        return self.config['ml']

    def get_ml_model_path(self) -> Path:
        return Path(self.get_ml()['output_dir']) / 'models'

    def get_ml_predictions_path(self) -> Path:
        return Path(self.get_ml()['output_dir']) / 'predictions'

    def get_ml_stats_path(self) -> Path:
        return Path(self.get_ml()['output_dir']) / 'stats'

    def get_ml_params(self) -> dict:
        return self.get_ml()["params"]

    def get_method_params(self) -> dict:
        return self.get_ml_params()["method"]
