import torch
import os
import yaml

def load_config_data(config_path='./configs/v1-inference.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def get_unique_prefixes(state_dict):
    return set(key.split('.'))[0] for key in state_dict.keys())

def analyze_block_structure(state_dict, block_name, depth=None):
    block_keys = [key for key in state_dict if key.startswith(block_name)]

    block_structure = {}
    for key in block_keys:
        layers = key.split('.')[len(block_name.split('.')):]

        if depth is not None:
            layers = layers[:depth]

        layer_name = '.'.join(layers)
        block_structure[layer_name] = block_structure.get(layer_name, 0) + 1

    return block_structure

def print_block_structure(block_structure):
    for layer, count in block_structure.items():
        print(f"{layer}: {count} keys")
