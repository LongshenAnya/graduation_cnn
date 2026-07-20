"""环境信息检测脚本"""

import platform
import torch
import subprocess
import psutil  # 需要安装：pip install psutil

def get_gpu_info():
    """获取GPU信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{gpu_name} ({gpu_memory:.0f}GB)"
    return "None (CUDA不可用)"

def get_cpu_info():
    """获取CPU信息"""
    if platform.system() == "Windows":
        # Windows
        try:
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                return processor.Name
        except:
            return platform.processor()
    else:
        # Linux/Mac
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except:
            return platform.processor()
    return "Unknown"

def get_memory_info():
    """获取内存信息"""
    mem = psutil.virtual_memory()
    return f"{mem.total / 1024**3:.0f}GB"

def main():
    print("=" * 60)
    print("实验环境信息")
    print("=" * 60)
    
    print("\n【硬件环境】")
    print(f"GPU: {get_gpu_info()}")
    print(f"CPU: {get_cpu_info()}")
    print(f"内存: {get_memory_info()}")
    
    print("\n【软件环境】")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA (PyTorch): {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    # 查看CUDA驱动版本
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"NVIDIA驱动: {result.stdout.strip()}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()