import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object, \
    save_numpy_array_data, evaluate_model
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
            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'LogisticRegression': LogisticRegression(),
                'SVC': SVC(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'GaussianNB': GaussianNB(),
                'MultinomialNB': MultinomialNB(),
                'BernoulliNB': BernoulliNB(),
                'XGBoost': xgb.XGBClassifier(),
                'LightGBM': lgb.LGBMClassifier(),
                'CatBoost': cb.CatBoostClassifier()
            }

            # 定义模型训练的超参数
            # 建议的超参数搜索空间
            # 可以根据实际需求和计算资源进行调整

            params = {
                'RandomForestClassifier': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 8]
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'LogisticRegression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']  # liblinear 支持 l1 和 l2，saga 支持 l1, l2, elasticnet
                },
                'SVC': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1: 曼哈顿距离, 2: 欧氏距离
                },
                'GaussianNB': {
                    # 高斯朴素贝叶斯通常不需要太多调参
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
                'MultinomialNB': {
                    'alpha': [0.1, 0.5, 1.0]  # 平滑参数
                },
                'BernoulliNB': {
                    'alpha': [0.1, 0.5, 1.0]  # 平滑参数
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'max_depth': [-1, 10, 20]
                },
                'CatBoost': {
                    # CatBoost 通常在默认参数下表现就很好，但也可以微调
                    'iterations': [200, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8]
                }
            }
            # 使用训练函数
            model_report:dict= evaluate_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                              models=models,params=params)

            # 获取最佳模型
            best_model_score = max(sorted(model_report.values()))

            # 获取最佳模型的名称
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            y_train_pred = best_model.predict(x_train)

            classification_train_metric = get_classification_metric(y_true=y_train,y_pred=y_train_pred)

            # 追踪MLFlow
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_metric(y_true=y_test,y_pred=y_test_pred)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            # 保存模型
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_Model)

            # 创建一个模型训练组件
            ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=classification_train_metric,
                                 test_metric_artifact=classification_test_metric)

            logging.info(f"模型训练完成，最佳模型：{best_model_name}，得分：{best_model_score},"
                         f"生成的模型文件路径：{self.model_trainer_config.trained_model_file_path}")

            return ModelTrainerArtifact

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


