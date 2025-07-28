from fastapi import APIRouter
from ..models.base import APIResponse
from ..services.visibility_service import get_visibility_mock_data

router = APIRouter(tags=["仿真选择-能见度"], prefix="/simulation/visibility")

@router.get("/data", response_model=APIResponse)
def get_visibility_data():
    """获取能见度数据接口"""
    # TODO: 实现实际业务逻辑
    mock_data = get_visibility_mock_data()
    return APIResponse(data=mock_data)