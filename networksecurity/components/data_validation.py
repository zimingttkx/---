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
import os, sys
from networksecurity.entity.artifact_entity import DataValidationArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validation_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"Schema 中要求的列数: {number_of_columns}")
            logging.info(f"当前数据中的列数: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def is_numerical_column_exists(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []

            for col in numerical_columns:
                if col not in dataframe_columns:
                    logging.info(f"必需的数值列 [{col}] 在 DataFrame 中缺失。")
                    return False
                if dataframe[col].dtype not in ['int64', 'float64']:
                    missing_numerical_columns.append(col)

            if len(missing_numerical_columns) == 0:
                return True
            else:
                logging.info(f"以下数值列的数据类型不正确: {missing_numerical_columns}")
                return False

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if is_same_dist.pvalue <= threshold:
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

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("开始数据验证流程")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            logging.info("验证训练集和测试集的列数...")
            status = self.validation_number_of_columns(dataframe=train_dataframe)
            if not status:
                raise Exception("训练数据列数与 Schema 不匹配")

            status = self.validation_number_of_columns(dataframe=test_dataframe)
            if not status:
                raise Exception("测试数据列数与 Schema 不匹配")

            logging.info("验证训练集和测试集的数值列是否存在且类型正确...")
            status = self.is_numerical_column_exists(dataframe=train_dataframe)
            if not status:
                raise Exception("训练数据的数值列存在问题")

            status = self.is_numerical_column_exists(dataframe=test_dataframe)
            if not status:
                raise Exception("测试数据的数值列存在问题")

            logging.info("检测数据集之间是否存在数据漂移...")
            # 注意：此处应比较训练集和测试集，以确保它们来自相似的分布
            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            validation_status = drift_status  # 最终的验证状态取决于所有检查
            if not validation_status:
                logging.warning("数据验证失败，发现数据漂移或结构问题。")
                # 这里可以根据需要决定是否要抛出异常停止流程
            else:
                logging.info("数据验证成功！正在将验证后的数据写入新目录...")
                # 创建存放验证后数据的目录
                valid_data_dir = os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(valid_data_dir, exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)

                # --- 【修复】 ---
                # 修正了保存测试集时使用错误路径的问题
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
                logging.info(f"已将验证后的训练集保存至: {self.data_validation_config.valid_train_file_path}")
                logging.info(f"已将验证后的测试集保存至: {self.data_validation_config.valid_test_file_path}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,  # 假设没有专门处理无效文件的流程
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"数据验证产出物创建成功: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

