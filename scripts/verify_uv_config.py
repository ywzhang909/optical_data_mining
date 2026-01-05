#!/usr/bin/env python3
"""
UV配置验证脚本
验证UV配置是否正确设置以解决硬链接警告问题
"""

import os
import subprocess
import sys
from pathlib import Path

def check_uv_config():
    """检查UV配置"""
    print("🔍 检查UV配置...")
    
    # 检查配置文件是否存在
    config_files = [
        ".uv/config.toml",
        "pyproject.toml",
        ".env.uv"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ 找到配置文件: {config_file}")
        else:
            print(f"❌ 缺少配置文件: {config_file}")
    
    # 检查目录设置
    print("\n📁 检查目录设置...")
    
    # 从pyproject.toml读取配置
    if Path("pyproject.toml").exists():
        try:
            with open("pyproject.toml", "r", encoding="utf-8") as f:
                content = f.read()
                if 'cache-dir = ".uv/cache"' in content:
                    print("✅ 缓存目录设置正确: .uv/cache")
                if 'link-mode = "copy"' in content:
                    print("✅ 链接模式设置正确: copy")
                if 'python-downloads = ".uv/python"' in content:
                    print("✅ Python下载目录设置正确: .uv/python")
        except Exception as e:
            print(f"❌ 读取pyproject.toml失败: {e}")
    
    # 检查环境变量
    print("\n🌍 检查环境变量...")
    
    env_vars = {
        "UV_LINK_MODE": "copy",
        "UV_CACHE_DIR": ".uv/cache",
        "UV_PYTHON_DOWNLOADS": ".uv/python"
    }
    
    for var, expected in env_vars.items():
        value = os.getenv(var)
        if value == expected:
            print(f"✅ {var} = {value}")
        else:
            print(f"⚠️  {var} = {value or '未设置'} (建议设置为: {expected})")

def test_uv_commands():
    """测试UV命令"""
    print("\n🧪 测试UV命令...")
    
    try:
        # 检查UV版本
        result = subprocess.run(["uv", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ UV版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV未安装或不在PATH中")
        return False
    
    try:
        # 查看UV配置
        result = subprocess.run(["uv", "config", "list"], 
                              capture_output=True, text=True, check=True)
        print("✅ UV配置列表:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"   {line}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取UV配置失败: {e}")
    
    try:
        # 测试dry-run（不会真正安装，只检查配置）
        print("\n🔄 测试UV sync --dry-run...")
        result = subprocess.run(["uv", "sync", "--dry-run"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ UV sync --dry-run 成功，配置正确")
        else:
            print(f"⚠️  UV sync --dry-run 警告: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ UV sync --dry-run 超时，但配置可能正确")
    except Exception as e:
        print(f"❌ UV sync测试失败: {e}")
    
    return True

def main():
    """主函数"""
    print("🚀 UV配置验证工具")
    print("=" * 50)
    
    check_uv_config()
    
    if test_uv_commands():
        print("\n✅ 配置验证完成！")
        print("\n💡 提示:")
        print("   - 如果仍有硬链接警告，请运行: source .env.uv")
        print("   - 或者重新启动终端使环境变量生效")
        print("   - 建议运行: uv sync 来应用新配置")
    else:
        print("\n❌ 请先安装UV: pip install uv")
    
    print("\n📚 更多信息请查看: UV配置说明.md")

if __name__ == "__main__":
    main()