import logging
import os
from datetime import datetime

# 1. 定义日志文件名
LOG_FILE_NAME = f"networksecurity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 2. 定义日志文件夹的路径
LOGS_DIRECTORY = os.path.join(os.getcwd(), "logs")

# 3. 确保日志文件夹存在
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# 4. 定义最终的、完整的日志文件路径
LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

# 5. 配置日志记录器
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding= "utf-8"
)