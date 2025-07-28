import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
mock_data_dir = os.path.join(current_dir, '../mock_data')

def get_light_source_mock_data():
    """读取光源数据的模拟数据"""
    file_path = os.path.join(mock_data_dir, 'light_source_data.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
