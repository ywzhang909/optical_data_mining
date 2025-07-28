import json
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接 mock_data 文件夹路径
mock_data_dir = os.path.join(current_dir, '../mock_data')

def get_cn2_device_mock_data():
    """读取 CN2 设备数据的模拟数据"""
    file_path = os.path.join(mock_data_dir, 'cn2_device_data.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cn2_mean_mock_data():
    """读取 CN2 均值数据的模拟数据"""
    file_path = os.path.join(mock_data_dir, 'cn2_mean.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
