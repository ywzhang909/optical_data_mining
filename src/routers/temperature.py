from fastapi import APIRouter
from ..models.base import APIResponse
from ..services.temperature_service import get_temperature_mock_data

router = APIRouter(tags=["仿真选择-温度"], prefix="/simulation/temperature")

@router.get("/data", response_model=APIResponse)
def get_temperature_data():
    """获取温度数据接口"""
    # TODO: 实现实际业务逻辑
    mock_data = get_temperature_mock_data()
    return APIResponse(data=mock_data)