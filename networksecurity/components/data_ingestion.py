import logging

from pymongo.synchronous.collection import Collection

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging import logger

# 数据摄取配置
from networksecurity.entity.config_entity import DataIngestionConfig
import os
import sys
import pymongo
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split


# 首先读取数据
from dotenv import load_dotenv

from push_data import MONGO_DB_URL

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion():
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def export_collection_as_dataframe(self):
        """
        从MongoDB中读取数据并导出为DataFrame
        :return: 处理好的DataFrame
        """
        try:
            # 连接到MongoDB数据库
            database_name = self.data_ingestion_config.database_name
            # 获取集合名称
            collection_name = self.data_ingestion_config.collection_name
            # 创建MongoDB客户端
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            # 获取指定的数据库和集合
            collection = self.mongo_client[database_name][collection_name]

            # 将MongoDB集合导出为DataFrame
            df = pd.DataFrame(list(collection.find()))
            # 检查是否有_id列，如果有则删除
            if "_id" in df.columns:
                df = df.drop("_id",axis = 1)
            # 将数据集中所有的"na"值替换为NaN
            df.replace({"na":np.nan},inplace = True)
            return df
        except Exception as e:
            raise NetworkSecurityException(sys,e)


    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        """
        将数据集导出到特征存储中
        :param dataframe:
        :return:
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # 创建文件夹
            dir_name = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_name,exist_ok = True)
            # 将DataFram转化为csv文件
            dataframe.to_csv(feature_store_file_path,index = False)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(sys,e)


    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        """
        将数据集拆分为训练集和测试集
        :param dataframe:
        :return:
        """
        try:
            train_set,test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)

            logging.info("训练集和测试集已经拆分")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info("训练集和测试集已经保存到本地")

            # 将训练集和测试集保存到特定的文件夹
            train_set.to_csv(self.data_ingestion_config.train_file_path,index = False)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index = False)

        except Exception as e:
            raise NetworkSecurityException(sys,e)


    def initiate_data_ingestion(self):
        """
        数据摄取的主函数
        :return:
        """
        try:
            dataframe = self.export_collection_as_dataframe
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
        except Exception as e:
            raise NetworkSecurityException(sys,e)
