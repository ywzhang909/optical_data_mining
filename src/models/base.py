from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, List, Dict, Any

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """API统一响应模型"""
    code: int = 200
    message: bool = True
    data: Optional[T] = None

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": True,
                "data": {}
            }
        }

class EmptyResponse(APIResponse[None]):
    """无数据响应模型"""
    data: None = None

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": True,
                "data": None
            }
        }