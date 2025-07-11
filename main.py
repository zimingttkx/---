from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, \
    ModelTrainerConfig
from networksecurity.logging.logger import logging
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_transformation import DataTransformation


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("数据摄取开始")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("数据摄取完成")
        print(dataingestionartifact)
        print("-------------------------------------------------------")
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("数据验证开始")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("数据验证完成")
        print(data_validation_artifact)
        print("-------------------------------------------------------")
        logging.info("数据转换开始配置")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("数据转换配置成功,开始进行数据转化")
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("数据转换开始")
        data_transformed_artifact = data_transformation.initiate_data_transformation()
        logging.info("数据转换完成")
        print(data_transformed_artifact)
        print("-------------------------------------------------------")
        #data_transformation.initiate_data_transformation()
        logging.info("数据转化完成 开始进行模型训练")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer= ModelTrainer(model_trainer_config,data_transformed_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("模型训练完成 超参数调正完毕")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
