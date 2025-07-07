# 这是数据验证的模块
from networksecurity.entity.artifact_entity import DataIngestionArtifact
# 这是需要的初始化信息
from networksecurity.entity.config_entity import DataValidationConfig
# 导入异常处理模块
from networksecurity.exception.exception import NetworkSecurityException
# 导入日志记录模块
from networksecurity.logging.logger import logging
# 导入用于检查是否发生数据飘逸的模块
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from networksecurity.entity.artifact_entity import DataValidationArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_file= read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

