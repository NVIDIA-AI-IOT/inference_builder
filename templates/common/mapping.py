from typing import List, Dict
from jinja2 import Template
from common.utils import get_logger

logger = get_logger(__name__)

class InputMapping:
    def __init__(self, required: List, giving: Dict, i_map: Dict):
        self._required = required
        self._giving = giving
        self._map = i_map

    def __call__(self):
        mapped = dict()
        for item in self._required:
            name = item['name']
            if name in self._map:
                # TODO template support
                tpl_str = self._map[name]
                value = Template(tpl_str).render(data=self._giving)
                mapped[name] = value
            elif name in self._giving:
                value = self._giving[name]
                mapped[name] = value
            else:
                logger.warning(f"Missing input: {name}")
        return mapped
