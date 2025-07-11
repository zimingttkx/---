import os
import sys

from mlflow.metrics import f1_score
from sklearn.metrics import precision_score

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

# 导入mlflow和dagshub进行云端管理整个实验
import mlflow
import joblib  # 导入 joblib 用于保存模型
import dagshub


# 这样初始化之后实验文件将不会存在于本地
dagshub.init(repo_owner='zimingttkx', repo_name='---', mlflow=True)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def tract_mlflow(self, best_model, classificationmetric):
        """
        【方案 A】
        记录MLflow的指标和模型到 DagsHub。
        此版本使用 log_artifact 来兼容 DagsHub。
        """
        try:
            # 开始一个新的MLflow运行
            with mlflow.start_run():
                # -------------------------------------------------------------
                #  记录指标 (这部分代码是正确的，予以保留)
                # -------------------------------------------------------------
                f1_score = classificationmetric.f1_score
                precision_score = classificationmetric.precision_score
                recall_score = classificationmetric.recall_score

                print(f"记录指标: F1={f1_score:.4f}, Precision={precision_score:.4f}, Recall={recall_score:.4f}")
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision_score)
                mlflow.log_metric("recall_score", recall_score)

                # -------------------------------------------------------------
                #  记录模型 (使用 joblib + log_artifact 替换 log_model)
                # -------------------------------------------------------------
                print("正在将模型保存为工件 (artifact)...")

                # 1. 定义一个本地临时路径用于存放模型文件
                local_model_dir = "final_models"
                os.makedirs(local_model_dir, exist_ok=True)
                local_model_path = os.path.join(local_model_dir, "model.pkl")

                # 2. 使用 joblib 将模型对象序列化到本地文件
                joblib.dump(best_model, local_model_path)

                # 3. 使用 log_artifact 将本地模型文件上传到 MLflow
                # "model_artifacts" 是您在 MLflow UI 的 Artifacts 中看到的文件夹名称
                mlflow.log_artifact(local_model_path, artifact_path="model_artifacts")

                print("模型已作为工件成功记录。")

        except Exception as e:
            # 抛出网络安全异常
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
                    'n_estimators': [50, 100],  # 减少树的数量
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.5, 1.0]
                },
                'LogisticRegression': {
                    # liblinear 对小数据集收敛快，且同时支持 L1 和 L2
                    'C': [0.1, 1],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'SVC': {
                    # rbf 是最常用的核，poly 计算成本较高，初期可省略
                    'C': [1, 10],
                    'gamma': ['scale'],
                    'kernel': ['rbf']
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                'GaussianNB': {
                    # 此模型训练极快，参数影响有限，可保留原样
                    'var_smoothing': [1e-9, 1e-8]
                },
                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'colsample_bytree': [0.8]  # 固定一个常用值
                },
                'LightGBM': {
                    'n_estimators': [50],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [50]
                },
                'CatBoost': {
                    # CatBoost 加上 verbose=0 参数可以在训练中保持静默
                    'iterations': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'depth': [4, 6]
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

            # 跟踪训练流程
            self.tract_mlflow(best_model,classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"最佳模型在 训练集 上的指标: {classification_train_metric}")
            logging.info(f"最佳模型在 测试集 上的指标: {classification_test_metric}")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)
            # 将最后的模型保存到文件夹
            save_object("final_models/model.pkl",best_model)
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