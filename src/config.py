import json


class AppConfig(object):
    def __init__(self, config_file: str = '../config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get_input(self) -> dict:
        return self.config['input']

    def get_log(self) -> dict:
        return self.config['log']

    def get_embedding_size(self) -> int:
        return self.config['embedding_size']

    def get_ml(self) -> dict:
        return self.config['ml']
