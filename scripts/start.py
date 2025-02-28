"""启动脚本"""
import os
import sys
import time
import platform
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录
os.chdir(str(project_root))

def set_default_environment():
    """设置默认环境变量"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        print("📝 从.env文件加载配置...")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print("ℹ️ 未找到.env文件，将使用默认配置")

def run_config_check() -> bool:
    """运行配置检查
    
    Returns:
        bool: 是否检查通过
    """
    try:
        # 导入配置检查模块
        check_script = Path(__file__).parent / "check_config.py"
        result = subprocess.run(
            [sys.executable, str(check_script)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # 打印输出
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"[错误] 配置检查失败: {str(e)}")
        return False

def check_and_kill_port(port: int):
    """检查并结束占用端口的进程
    
    Args:
        port: 端口号
    """
    try:
        # 查找占用端口的进程
        if platform.system() == "Windows":
            result = subprocess.run(
                f"netstat -ano | findstr :{port}",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    if f":{port}" in line:
                        pid = line.strip().split()[-1]
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                        print(f"[成功] 结束了占用端口 {port} 的进程 {pid}")
        else:
            result = subprocess.run(
                f"lsof -i :{port} | grep LISTEN",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    pid = line.split()[1]
                    subprocess.run(f"kill -9 {pid}", shell=True)
                    print(f"[成功] 结束了占用端口 {port} 的进程 {pid}")
    except Exception as e:
        print(f"[警告] 检查端口占用失败: {str(e)}")

def start_backend():
    """启动后端服务"""
    try:
        print("[后端] 启动后端服务...")
        # 检查端口占用
        check_and_kill_port(8000)
        
        backend_process = subprocess.Popen(
            ["python", "-m", "uvicorn", "src.api.main:app", "--reload"],
            cwd=project_root
        )
        print("[成功] 后端服务启动成功")
        return backend_process
    except Exception as e:
        print(f"[错误] 后端服务启动失败: {str(e)}")
        return None

def start_frontend():
    """启动前端服务"""
    frontend_dir = project_root / "frontend"
    if not frontend_dir.exists():
        print("[错误] 前端目录不存在")
        return None
        
    try:
        print("[前端] 启动前端服务...")
        # 检查端口占用
        check_and_kill_port(3000)
        
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=str(frontend_dir),
            shell=platform.system() == "Windows"
        )
        print("[成功] 前端服务启动成功")
        return frontend_process
    except Exception as e:
        print(f"[错误] 前端服务启动失败: {str(e)}")
        return None

def stop_services(backend_process, frontend_process):
    """停止所有服务
    
    Args:
        backend_process: 后端进程
        frontend_process: 前端进程
    """
    if backend_process:
        backend_process.terminate()
        backend_process.wait()
    
    if frontend_process:
        frontend_process.terminate()
        frontend_process.wait()
        
    # 确保端口被释放
    check_and_kill_port(3000)
    check_and_kill_port(8000)

def main():
    """主函数"""
    print("[开始] 开始启动服务...")
    
    # 运行配置检查
    if not run_config_check():
        print("[错误] 配置检查失败，请修复问题后重试")
        sys.exit(1)
    
    print("[成功] 配置检查通过，开始启动服务...")
    
    # 启动后端服务
    backend_process = start_backend()
    if not backend_process:
        print("[错误] 后端服务启动失败")
        sys.exit(1)
    
    # 等待后端启动
    time.sleep(2)
    
    # 启动前端服务
    frontend_process = start_frontend()
    if not frontend_process:
        print("[错误] 前端服务启动失败")
        stop_services(backend_process, None)
        sys.exit(1)
    
    print("\n[完成] 服务启动完成！")
    print("[文档] API文档:")
    print("- Swagger UI: http://localhost:8000/docs")
    print("- ReDoc: http://localhost:8000/redoc")
    print("[页面] 前端页面: http://localhost:3000")
    
    try:
        # 等待任意一个进程结束
        while True:
            if backend_process.poll() is not None:
                print("[错误] 后端服务意外停止")
                break
            if frontend_process.poll() is not None:
                print("[错误] 前端服务意外停止")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[停止] 正在停止服务...")
    finally:
        stop_services(backend_process, frontend_process)
        print("[完成] 服务已停止")

if __name__ == "__main__":
    main() 