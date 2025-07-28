from fastapi import APIRouter
from ..models.base import APIResponse
from ..services.light_source_service import get_light_source_mock_data

router = APIRouter(tags=["仿真选择-光源"], prefix="/simulation/light-source")

@router.get("/data", response_model=APIResponse)
def get_light_source_data():
    """获取光源数据接口"""
    # TODO: 实现实际业务逻辑
    mock_data = get_light_source_mock_data()
    return APIResponse(data=mock_data)