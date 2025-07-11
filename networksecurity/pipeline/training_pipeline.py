"""
确保从数据摄取 数据转换 模型验证 模型评估 模型推送者

"""
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
TrainingPipelineConfig,DataIngestionConfig,
DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
DataIngestionArtifact,DataValidationArtifact,
DataTransformationArtifact,ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config= TrainingPipeline()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config= DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("开始进行数据摄取")


        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

