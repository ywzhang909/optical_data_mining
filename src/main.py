from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import light_source, cn2, wind_speed, temperature, absorption_coefficient, visibility

app = FastAPI(title="仿真数据API服务", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(light_source.router)
app.include_router(cn2.router)
app.include_router(wind_speed.router)
app.include_router(temperature.router)
app.include_router(absorption_coefficient.router)
app.include_router(visibility.router)

@app.get("/health", tags=["系统状态"])
def health_check():
    return {"status": "healthy", "message": "API服务运行正常"}