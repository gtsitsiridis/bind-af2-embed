from __future__ import annotations

import json
from enum import Enum
from pathlib import Path


class MethodName(Enum):
    EMBEDDINGS = "EMBEDDINGS"
    DISTMAPS = "DISTMAPS"
    COMBINED_V1 = "COMBINED_V1"
    COMBINED_V2 = "COMBINED_V2"


class RunTemplate(object):
    def __init__(self, template_path: Path):
        with open(str(template_path), 'r') as f:
            _template = json.load(f)
            self._params = _template['params']
            self._method_name = _template['method']
            self._name = template_path.stem

    @property
    def method_name(self) -> MethodName:
        return MethodName[self._method_name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def train_params(self) -> dict:
        return self._params['train']

    @property
    def model_params(self) -> dict:
        return self._params['model']

    @property
    def optimizer_params(self) -> dict:
        return self._params['optimizer']

    @property
    def loss_params(self) -> dict:
        return self._params['loss']

    def __str__(self):
        return f'Template {self._method_name}: {str(self._params)}'
