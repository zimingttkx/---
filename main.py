from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from networksecurity.logging.logger import logging
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception import NetworkSecurityException
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

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
