from fastapi import APIRouter
from ..models.base import APIResponse
from ..services.wind_speed_service import get_wind_speed_mock_data

router = APIRouter(tags=["仿真选择-风速"], prefix="/simulation/wind-speed")

@router.get("/data", response_model=APIResponse)
def get_wind_speed_data():
    """获取风速数据接口"""
    # TODO: 实现实际业务逻辑
    mock_data = get_wind_speed_mock_data()
    return APIResponse(data=mock_data)