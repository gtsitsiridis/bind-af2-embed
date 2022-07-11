import json
from pathlib import Path
from typing import Generator

from ml.template import RunTemplate
import os


class InputConfig(object):
    def __init__(self, config: dict):
        self._config = config

    @property
    def params(self) -> dict:
        return self._config['params']

    @property
    def files(self) -> dict:
        return self._config['files']


class OutputConfig(object):
    def __init__(self, path: Path):
        self._path = path

    @property
    def ml_path(self) -> Path:
        return Path(self._path) / 'ml'

    @property
    def model_path(self) -> Path:
        return self.ml_path / 'models'

    @property
    def predictions_path(self) -> Path:
        return self.ml_path / 'predictions'

    @property
    def stats_path(self) -> Path:
        return self.ml_path / 'stats'


class LoggingConfig(object):
    def __init__(self, config: dict):
        self._tracemalloc = config.get('tracemalloc', False)
        self._path = config['path']
        self._loggers = config['loggers']

    @property
    def tracemalloc(self) -> bool:
        return self._tracemalloc

    @property
    def path(self) -> str:
        return self._path

    @property
    def loggers(self) -> dict:
        return self._loggers


class AppConfig(object):
    def __init__(self, config_file: str = None):
        if config_file is None:
            config_file = '../config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)
            self._input = InputConfig(config['input'])
            self._output = OutputConfig(Path(config['output']))
            self._templates_path = Path(config['templates'])
            self._log = LoggingConfig(config['log'])

    @property
    def input(self) -> InputConfig:
        return self._input

    @property
    def log(self) -> LoggingConfig:
        return self._log

    @property
    def output(self) -> OutputConfig:
        return self._output

    def resolve_template(self, template_name: str) -> RunTemplate:
        template_path = self._templates_path / (template_name + '.json')
        assert template_path.exists(), f'The template {str(template_name)} can not be found!'
        return RunTemplate(template_path)

    def iter_templates(self):
        template_path = self._templates_path
        for file in template_path.iterdir():
            yield self.resolve_template(file.stem)
