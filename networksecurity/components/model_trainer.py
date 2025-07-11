import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, \
    ClassificationMetricArtifact
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'LogisticRegression': LogisticRegression(),
                'SVC': SVC(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'GaussianNB': GaussianNB(),
                'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(),
                'CatBoost': cb.CatBoostClassifier(verbose=False, allow_writing_files=False)
            }

            params = {
                'RandomForestClassifier': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 8],
                    'subsample': [0.7, 0.8, 0.9]
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                    # 'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)] # 进阶调优
                },
                'LogisticRegression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'SVC': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly']
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1: 曼哈顿距离 (Manhattan), 2: 欧氏距离 (Euclidean)
                },
                'GaussianNB': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 70],
                    'max_depth': [-1, 10, 20],
                    'colsample_bytree': [0.7, 0.8]
                },
                'CatBoost': {
                    'iterations': [200, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5, 7]  # L2 正则化，防止过拟合
                }
            }

            model_report = {}
            best_estimators = {}
            best_params_report = {}

            for model_name, model in models.items():
                logging.info(f"====== 开始训练模型: {model_name} ======")
                param_grid = params.get(model_name, {})

                # <<< --- 核心修改：将 n_jobs 设为 1，实现顺序执行 --- >>>
                # verbose=2 可以让您看到每个参数组合的训练过程
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=1, verbose=2)

                grid_search.fit(x_train, y_train)

                model_report[model_name] = grid_search.best_score_
                best_estimators[model_name] = grid_search.best_estimator_
                best_params_report[model_name] = grid_search.best_params_

                logging.info(f"====== 模型 {model_name} 训练完成. 交叉验证分数: {grid_search.best_score_:.4f} ======\n")

            best_model_name = max(model_report, key=model_report.get)
            best_model = best_estimators[best_model_name]
            best_model_score = model_report[best_model_name]
            best_model_params = best_params_report[best_model_name]

            logging.info("=" * 40)
            logging.info("=== 所有模型训练完毕. 最终评估结果 ===")
            logging.info(f"==> 最佳模型: '{best_model_name}'")
            logging.info(f"==> 交叉验证得分: {best_model_score:.4f}")
            logging.info(f"==> 最佳参数: {best_model_params}")
            logging.info("=" * 40)

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"最佳模型性能 {best_model_score:.4f} 未达到预期阈值 {self.model_trainer_config.expected_accuracy}")

            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"最佳模型在 训练集 上的指标: {classification_train_metric}")
            logging.info(f"最佳模型在 测试集 上的指标: {classification_test_metric}")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"模型训练产出物成功创建: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("开始加载转换后的数据...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("数据加载完毕，即将开始模型训练...")
            model_trainer_artifact = self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e