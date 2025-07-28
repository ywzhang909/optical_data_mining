from fastapi import APIRouter
from typing import List
from ..models.base import APIResponse
from ..models.data_models import PaginatedRequest, DateRangeRequest, AbsorptionCoefficientDataItem, CN2MeanData
from ..services.absorption_coefficient_service import (
    get_absorption_coefficient_device_mock_data,
    get_absorption_coefficient_mean_mock_data
)

router = APIRouter(tags=["仿真选择-吸收系数"], prefix="/simulation/absorption-coefficient")

@router.post("/device-data", response_model=APIResponse[List[AbsorptionCoefficientDataItem]])
def get_absorption_coefficient_device_data(request: PaginatedRequest):
    """
    读取吸收系数设备数据接口
    根据日期范围和分页参数获取吸收系数仿真的设备数据
    """
    # TODO: 实现实际业务逻辑
    # 从 JSON 文件读取模拟数据
    mock_data = get_absorption_coefficient_device_mock_data()
    return APIResponse(data=mock_data)

@router.post("/mean", response_model=APIResponse[CN2MeanData])
def get_absorption_coefficient_mean(request: DateRangeRequest):
    """
    获取吸收系数均值数据接口
    根据日期范围获取吸收系数仿真的均值数据
    """
    # TODO: 实现实际业务逻辑
    # 从 JSON 文件读取模拟数据
    mock_data = get_absorption_coefficient_mean_mock_data()
    return APIResponse(data=mock_data)