from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_light_source_device_info():
    """测试光源参数-读取设备信息接口"""
    response = client.post(
        "/light-source/device-info",
        json={
            "date": "2025-07-19",
            "path_dir": "C:\\Users\\liudonghua\\Desktop\\GDS\\project\\simulationAlgorithm\\temporary_files\\2025-07-22\\张三\\2025-07-19"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["message"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert all(isinstance(item, dict) for item in data["data"])
    assert all("time" in item and "headline" in item for item in data["data"])