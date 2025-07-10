import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_numpy_array_data,load_object,save_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config= model_trainer_config
            self.data_transformation_artifact= data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e


    def train_model(self,x_train,y_train,x_test,y_test):
        """
        这个函数用于训练模型,并将模型保存到指定的目录中
        :param x_train:训练集的特征数据
        :param y_train:训练集里面的标签数据
        :return:返回一个训练好的模型
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        """
        这个函数用于训练模型，并将训练好的模型保存到指定的目录中。
        :return:返回一个训练好的模型 用于训练
        """
        try:
            logging.info("获取数据路径")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            logging.info("开始加载array数据")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            # 将拼好后的数据分割开来匹配到各自的数据集
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e


