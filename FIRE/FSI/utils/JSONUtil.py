import json
from typing import Dict, List


class JSONUtil:
    @staticmethod
    def dump(data: Dict | List, save_path: str, indent: int = 4):
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def load(path: str) -> Dict | List:
        with open(path, 'r') as f:
            return json.load(f)
