from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

import os,sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        # 尝试将preprocessor和model赋值给self.preprocessor和self.model
        try:
            self.preprocessor= preprocessor
            self.model = model
        # 如果出现异常，则抛出NetworkSecurityException
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def predict(self,x):
        """

        :param x:
        :return:
        """
        try:
            # 对输入数据进行预处理
            x_transform = self.preprocessor.transform(x)
            # 使用模型对预处理后的数据进行预测
            y_hat = self.model.predict(x_transform)
            # 返回预测结果
            return y_hat
        except Exception as e:
            # 抛出自定义异常
            raise NetworkSecurityException(e,sys) from e