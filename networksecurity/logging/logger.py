import logging
import os
from datetime import datetime

# 设置日志文件名
LOG_FILE = f"networksecurity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 创建日志目录
logs_path = os.path.join(os.getcwd(), "logs",LOG_FILE)
# 确保日志目录存在
os.makedirs(logs_path, exist_ok=True)

# 设置日志文件的完整路径
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 配置日志记录器
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level= logging.INFO
)
