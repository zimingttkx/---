import os
import sys
import numpy as np
import pandas as pd

"""
训练流程中常用的常量定义
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME:str = "NetworkSecurity"
ARTIFACTS_DIR:str = "Artifacts"
FILE_NAME:str = "phisingData.csv"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

"""
1.数据摄取文件路径
2.特征存储路径
3.训练文件路径
4.测试文件路径
5.训练数据占比

"""
DATA_INGESTION_COLLECTION_NAME = "NetworkData"
DATA_INGESTION_DATABASE_NAME:str= "zimingttkx"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURES_STORE_DIR = "features_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2