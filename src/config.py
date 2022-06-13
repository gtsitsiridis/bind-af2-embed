import json


class AppConfig(object):
    def __init__(self, config_file: str = '../config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get_files(self) -> dict:
        return self.config['files']
