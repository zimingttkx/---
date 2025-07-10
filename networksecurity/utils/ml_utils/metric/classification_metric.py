from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score,precision_score,recall_score
from networksecurity.logging.logger import logging
import sys

def get_classification_metric(y_true,y_pred)-> ClassificationMetricArtifact:
    """
    这个函数用于计算分类模型的评估指标，包括F1分数、精确度和召回率。
    它接收真实值和预测值作为输入，并返回一个包含这些指标的
    ClassificationMetricArtifact 对象
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 创建的指标类
    """
    try:
        # 计算f1_score
        model_f1_score = f1_score(y_true,y_pred)
        # 计算precision_score
        model_precision_score = precision_score(y_true,y_pred)
        # 计算recall_score
        model_recall_score = recall_score(y_true,y_pred)
        # 创建ClassificationMetricArtifact对象
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score)
        # 返回ClassificationMetricArtifact对象
        logging.info(
            f"模型评估指标：f1_score={model_f1_score}, precision_score={model_precision_score}, recall_score={model_recall_score}")
        return classification_metric

    except Exception as e:
        # 抛出NetworkSecurityException异常
        raise NetworkSecurityException(e,sys) from e