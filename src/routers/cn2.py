from fastapi import APIRouter
from typing import List
from ..models.base import APIResponse
from ..models.data_models import PaginatedRequest, DateRangeRequest, CN2DataItem, CN2MeanData
from ..services.cn2_service import get_cn2_device_mock_data, get_cn2_mean_mock_data

router = APIRouter(tags=["仿真选择-cn2"], prefix="/simulation/cn2")

@router.post("/device-data", response_model=APIResponse[List[CN2DataItem]])
def get_cn2_device_data(request: PaginatedRequest):
    """
    读取CN2设备数据接口
    根据日期范围和分页参数获取CN2仿真的设备数据
    """
    # TODO: 实现实际业务逻辑
    # 从 JSON 文件读取模拟数据
    mock_data = get_cn2_device_mock_data()
    return APIResponse(data=mock_data)

@router.post("/mean", response_model=APIResponse[CN2MeanData])
def get_cn2_mean(request: DateRangeRequest):
    """
    获取CN2均值数据接口
    根据日期范围获取CN2仿真的均值数据
    """
    # TODO: 实现实际业务逻辑
    # 从 JSON 文件读取模拟数据
    mock_data = get_cn2_mean_mock_data()
    return APIResponse(data=mock_data)