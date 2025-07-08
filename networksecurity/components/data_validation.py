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
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config= read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e


    # 定义一个用来读取数据的函数 使用静态方法
    @staticmethod
    def read_data(file_path) ->pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e


    # 定义一个函数用来检查数据的行数是否缺失
    def validation_number_of_columns(self,dataframe:pd.DataFrame)-> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"正常数据的的列数为: {number_of_columns}")
            logging.info(f"实际数据的列数为: {len(dataframe.columns)}")
            ## 如果实际数据的列数与正常数据的列数相等，则返回True
            if len(dataframe.columns) == number_of_columns:
                logging.info("数据的列数与正常数据的列数相等")
                return True
            else:
                return False

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    # 定义一个函数用来检查行数据是否为数值类行
    def is_numerical_column_exists(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            # 标记是否存在缺失的数值列
            missing_numerical_columns = []

            for col in numerical_columns:
                if col not in dataframe_columns:
                    # 如果 schema 中定义的列在 DataFrame 中不存在，直接返回 False
                    logging.info(f"必需的列 [{col}] 在 DataFrame 中缺失。")
                    return False

                # 检查该列的数据类型是否为数值类型
                if dataframe[col].dtype not in ['int64', 'float64']:
                    missing_numerical_columns.append(col)

            if len(missing_numerical_columns) == 0:
                logging.info("所有在 schema 中定义的数值列都具有正确的数值类型。")
                return True
            else:
                logging.info(f"以下在 schema 中定义的数值列，其实际数据类型不正确: {missing_numerical_columns}")
                return False

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # 定义一个函数用来检查是否发生数据飘逸
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            # 状态默认为True (无漂移)，一旦发现漂移就变为False
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)

                # p-value > threshold 意味着分布相同 (无漂移)
                # p-value <= threshold 意味着分布不同 (有漂移)
                if is_same_dist.pvalue <= threshold:
                    # 发现一处漂移，则整体状态变为有漂移
                    status = False
                    same_distribution = False
                else:
                    same_distribution = True

                report.update(
                    {column: {"p_value": float(is_same_dist.pvalue), "same_distribution": same_distribution}}
                )

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            ## 直接调用上方定义的静态方法读取训练数据和测试数据
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # 检查训练数据和测试数据的列数是否相等
            status = self.validation_number_of_columns(dataframe = train_dataframe)
            if not status:
                raise Exception("训练数据的列数与正常数据的列数不相等")

            status = self.validation_number_of_columns(dataframe = test_dataframe)
            if not status:
                raise Exception("测试数据的列数与正常数据的列数不相等")

            # 检查训练数据和测试数据的数值类列数是否相等
            status = self.is_numerical_column_exists(dataframe = train_dataframe)
            if not status:
                raise Exception("训练数据的数值类列数与正常数据的数值类列数不相等")

            status = self.is_numerical_column_exists(dataframe = test_dataframe)
            if not status:
                raise Exception("测试数据的数值类列数与正常数据的数值类列数不相等")

            # 检查训练数据和测试数据是否发生数据飘逸
            status = self.detect_dataset_drift(base_df = train_dataframe,current_df = test_dataframe)
            if not status:
                raise Exception("训练数据和测试数据发生数据飘逸")
            else:
                dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path,exist_ok=True)
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
                test_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)

