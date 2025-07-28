import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
mock_data_dir = os.path.join(current_dir, '../mock_data')

def get_other_device_mock_data():
    """读取其他设备数据的模拟数据"""
    file_path = os.path.join(mock_data_dir, 'other_device_data.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_other_mean_mock_data():
    """读取其他均值数据的模拟数据"""
    file_path = os.path.join(mock_data_dir, 'other_mean.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
