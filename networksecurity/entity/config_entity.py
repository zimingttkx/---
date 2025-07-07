from networksecurity.constant import training_pipeline
from datetime import datetime
import os

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACTS_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACTS_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp,self.pipeline_name)
        self.timestamp = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
        training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_dir:str = os.path.join(training_pipeline_config.artifact_dir,
                                              training_pipeline.FEATURE_STORE_DIR_NAME)
        self.training_file_path:str = os.path.join(self.feature_store_dir,
                                               training_pipeline.TRAINING_FILE_NAME)
        self.testing_file_path:str = os.path.join(self.feature_store_dir,
                                               training_pipeline.TESTING_FILE_NAME)
        self.train_test_split_ratio:float = training_pipeline.TRAIN_TEST_SPLIT_RATIO
        self.collection_name:str = training_pipeline.COLLECTION_NAME
        self.database_name:str = training_pipeline.DATABASE_NAME