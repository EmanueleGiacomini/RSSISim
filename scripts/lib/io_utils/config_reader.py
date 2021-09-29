"""config_reader.py
"""

import argparse
import json


class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            config_json = json.load(f)
            for _k in config_json.keys():
                for k in config_json[_k].keys():
                    setattr(namespace, k, config_json[_k][k])
