from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date

# ------------------------------
# 通用数据模型
# ------------------------------
class DateRangeRequest(BaseModel):
    """日期范围请求模型"""
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", example="2025-07-22")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", example="2025-07-23")

class PaginatedRequest(DateRangeRequest):
    """带分页的请求模型"""
    page_num: int = Field(1, description="页码", ge=1, example=1)
    page_size: int = Field(6, description="每页数量", ge=1, le=100, example=6)

# ------------------------------
# 光源参数接口模型
# ------------------------------
class LightSourceRequest(BaseModel):
    """光源参数请求模型"""
    date: str = Field(..., description="历史实验日期 (YYYY-MM-DD)", example="2025-07-19")
    path_dir: str = Field(..., description="文件存放路径", example="C:\\Users\\liudonghua\\Desktop\\GDS\\project\\simulationAlgorithm\\temporary_files\\2025-07-22\\张三\\2025-07-19")

class LightSourceDataItem(BaseModel):
    """光源参数数据项"""
    time: str = Field(..., description="实验时间", example="14:00:00")
    headline: str = Field(..., description="实验标题名称", example="xx次实验")

# ------------------------------
# CN2接口模型
# ------------------------------
class CN2DataItem(BaseModel):
    """CN2数据项"""
    time: List[str] = Field(..., description="时间数组", example=["5:10", "5:20", "5:30", "5:40"])
    value: List[str] = Field(..., description="值数组", example=["5", "8", "3", "4"])

class CN2MeanData(BaseModel):
    """CN2均值数据"""
    distance: List[int] = Field(..., description="采集点", example=[100, 200, 300, 400, 500])
    value: List[str] = Field(..., description="值数组", example=["76", "64", "30", "73", "50"])

# ------------------------------
# 风速接口模型
# ------------------------------
class WindSpeedDataItem(BaseModel):
    """风速数据项"""
    time: List[str] = Field(..., description="时间数组", example=["5:10", "5:20", "5:30", "5:40"])
    value_x: List[str] = Field(..., description="x值数组", example=["5", "8", "3", "10"])
    value_y: List[str] = Field(..., description="y值数组", example=["6", "3", "12", "4"])
    value_z: List[str] = Field(..., description="z值数组", example=["3", "6", "3", "7"])

class WindSpeedMeanData(BaseModel):
    """风速均值数据"""
    distance: List[int] = Field(..., description="采集点", example=[100, 200, 300, 400, 500])
    vx_value: List[str] = Field(..., description="x方向风速值数组", example=["43", "87", "30", "43", "50"])
    vy_value: List[str] = Field(..., description="y方向风速值数组", example=["88", "56", "76", "43", "86"])
    vz_value: List[str] = Field(..., description="z方向风速值数组", example=["76", "65", "87", "23", "13"])"

# ------------------------------
# 温度接口模型
# ------------------------------
class TemperatureDataItem(BaseModel):
    """温度数据项"""
    time: List[str] = Field(..., description="时间数组", example=["5:10", "5:20", "5:30", "5:40"])
    value: List[str] = Field(..., description="温度值数组", example=["5", "8", "3", "4"])

class TemperatureMeanData(BaseModel):
    """温度均值数据"""
    distance: List[int] = Field(..., description="采集点", example=[100, 200, 300, 400, 500])
    value: List[str] = Field(..., description="温度均值数组", example=["76", "64", "30", "73", "50"])"

# ------------------------------
# 吸收系数接口模型
# ------------------------------
class AbsorptionCoefficientDataItem(BaseModel):
    """吸收系数数据项"""
    time: List[str] = Field(..., description="时间数组", example=["5:10", "5:20", "5:30", "5:40"])
    value: List[str] = Field(..., description="吸收系数值数组", example=["5", "8", "3", "4"])"

# ------------------------------
# 能见度接口模型
# ------------------------------
class VisibilityData(BaseModel):
    """能见度数据"""
    distance: List[int] = Field(..., description="采集点", example=[100, 200, 300, 400, 500])
    value: List[str] = Field(..., description="能见度值数组", example=["76", "64", "30", "73", "50"])"