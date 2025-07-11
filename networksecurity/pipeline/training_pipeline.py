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
        """
        开始数据摄取
        :return: 返回数据摄取组件
        """
        try:
            # 创建数据摄取配置
            self.data_ingestion_config= DataIngestionConfig(training_pipeline_config= self.training_pipeline_config)
            # 记录开始进行数据摄取
            logging.info("开始进行数据摄取")
            # 创建数据摄取对象
            data_ingestion= DataIngestion(data_ingestion= self.data_ingestion_config)
            # 初始化数据摄取
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            # 记录数据摄取完成
            logging.info(f"数据摄取完成,生成的Data_ingestion_artifact:{data_ingestion_artifact}")
            # 返回数据摄取结果
            return data_ingestion_artifact

        except Exception as e:
            # 抛出网络安全异常
            raise NetworkSecurityException(e,sys) from e

    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact):
        """
        这个函数用来进行数据验证
        :param data_ingestion_artifact:数据摄取函数的返回值 作为参数传入下一个数据处理步骤
        :return: 返回数据验证组件 Data_validation_artifact
        """
        try:
            # 根据训练管道配置初始化数据验证配置
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            # 根据数据摄取工件和数据验证配置初始化数据验证
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            # 记录初始化数据验证
            logging.info("初始化数据验证")
            # 调用数据验证的初始化方法
            data_validation_artifact = data_validation.initiate_data_validation()
            # 记录数据验证完成，并生成数据验证工件
            logging.info(f"数据验证完成,生成的Data_validation_artifact:{data_validation_artifact}")
            # 返回数据验证工件
            return data_validation_artifact

        except Exception as e:
            # 抛出网络安全异常
            raise NetworkSecurityException(e,sys) from e

    def start_data_transformation(self,data_validation_artifact: DataValidationArtifact):
        """
        这个函数用来进行数据转换
        :param data_validation_artifact: 数据验证组件作为返回值传给数据转换的参数
        :return: 返回数据转化的组件
        """
        try:
            # 创建数据转换配置
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            # 创建数据转换对象
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            # 记录开始进行数据转化的日志
            logging.info("开始进行数据转化")
            # 进行数据转化
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("数据转化完成")
            # 返回数据转化的组件
            return data_transformation_artifact

        except Exception as e:
            # 抛出网络安全异常
            raise NetworkSecurityException(e,sys) from e


    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact)->ModelTrainerArtifact:
        """
        这个函数用来返回模型训练组件
        :param data_transformation_artifact: 这个是数据转化的组件
        :return: 返回模型训练的组件
        """
        try:
            # 创建ModelTrainerConfig对象
            self.model_trainer_config:ModelTrainerConfig = ModelTrainerArtifact(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config)
            # 创建ModelTrainer对象
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config,
                                         data_transformation_artifact=data_transformation_artifact)

            # 初始化ModelTrainer对象
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            # 返回ModelTrainerArtifact对象
            return model_trainer_artifact

        except Exception as e:
            # 抛出NetworkSecurityException异常
            raise NetworkSecurityException(e,sys) from e


    def run_pipeline(self):
        """
        这个函数用来运行整个训练管道
        :return:
        """
        try:
            # 开始数据摄取
            data_ingestion_artifact = self.start_data_ingestion()
            # 开始数据验证
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # 开始数据转换
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # 开始模型训练
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            return model_trainer_artifact
        except Exception as e:
            # 抛出网络安全异常
            raise NetworkSecurityException(e,sys) from e