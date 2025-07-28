from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_cn2_device_data():
    """测试仿真选择-cn2-读取设备数据接口"""
    response = client.post(
        "/simulation/cn2/device-data",
        json={
            "start_date": "2025-07-22",
            "end_date": "2025-07-23",
            "page_num": 1,
            "page_size": 6
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["message"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 6
    assert all(isinstance(item, dict) for item in data["data"])
    assert all("time" in item and "value" in item for item in data["data"])

def test_cn2_mean():
    """测试仿真选择-cn2-均值接口"""
    response = client.post(
        "/simulation/cn2/mean",
        json={
            "start_date": "2025-07-22",
            "end_date": "2025-07-23"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["message"] is True
    assert isinstance(data["data"], dict)
    assert "distance" in data["data"] and "value" in data["data"]
    assert len(data["data"]["distance"]) == len(data["data"]["value"]) == 16