import os  # 用于处理文件和环境变量
import sys # 用于获取异常信息
from networksecurity.logging import logger # 用于日志记录
from networksecurity.exception import NetworkSecurityException # 用于自定义异常处理
import certifi # 用于处理SSL或者STL证书
import pandas as pd # 用于数据处理
import numpy as np # 用于数值计算
import pymongo # 用于连接MongoDB数据库
import json # 用于处理JSON数据
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


# 构建ETL管道
class NetworkDataExtractor():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    # 定义一个函数 将csv文件转化为json格式
    def cv_to_json(self,file_path):
        try:
            # 首先读取数据集
            data = pd.read_csv(file_path)
            # 删除没用的索引
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    # 定义一个函数 将数据集推送到MongoDB数据库
    def push_data_to_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]

            # 将数据集推送到MongoDB数据库
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__ == "__main__":
    # 定义数据集路径
    file_path = "Network_Data/phisingData.csv"
    database = "data"
    collection = "NetworkData"
    networkobj = NetworkDataExtractor()
    records = networkobj.cv_to_json(file_path)
    print(f"转化后的json数据是: {records}")  # 打印前5条记录以验证转换是否成功
    num_of_records = networkobj.push_data_to_mongodb(records,database,collection)
    print(f"成功推送{num_of_records}条数据到MongoDB数据库")