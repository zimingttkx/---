"""

这个文件用来写通用的函数或者方法 比如read_yaml_file等
"""

import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path:str) -> dict:
    """
    读取yaml文件
    :param file_path:  文件路径
    :return:  返回字典
    """
    try:
        # 打开yaml文件
        with open(file_path,"rb") as yaml_file:
            # 使用yaml.safe_load方法将文件内容转换为字典
            return yaml.safe_load(yaml_file)
    except Exception as e:
        # 抛出异常
        raise NetworkSecurityException(e,sys) from e

def write_yaml_file(file_path: str,content: object, replace: bool = False)-> None:
    """
    将内容写入yaml文件
    :param file_path:  文件路径
    :param content:  内容
    :param replace:  是否替换文件，默认为False
    :return:
    """
    try:
        # 如果replace为True，则删除文件
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        # 创建文件路径
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        # 打开文件，写入内容
        with open (file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        # 抛出异常
        raise NetworkSecurityException(sys,e) from e

def save_numpy_array_data(file_path: str,array: np.array):
    """
    保存numpy数组到指定路径的文件中
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok= True)
        # 将 "w" 修改为 "wb"，以二进制模式写入
        with open (file_path, "wb") as file_obj:
            np.save(file_obj,array)

    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def load_numpy_array_data(file_path: str)-> np.array:
    """
    加载numpy数组数据
    :param file_path:  文件路径
    :return:  numpy数组
    """
    try:
        # 打开文件
        with open(file_path,"rb") as file_obj:
            # 加载numpy数组
            return np.load(file_obj)
    except Exception as e:
        # 抛出异常
        raise NetworkSecurityException(e,sys) from e

def save_object(file_path: str,obj: object)-> None:
# 将对象保存到文件中
    try:

        logging.info(f"文件开始保存在路径 {file_path}")

        # 创建文件路径的目录
        os.makedirs(os.path.dirname(file_path),exist_ok= True)

        # 打开文件，以二进制写入模式
        with open(file_path,"wb") as file_obj:

            # 将对象保存到文件中
            pickle.dump(obj,file_obj)

        logging.info(f"文件保存成功 {file_path}")
    except Exception as e:

        # 抛出异常
        raise NetworkSecurityException(e,sys) from e

def load_object(file_path: str)-> object:
    """
    这个文件用于加载pkl文件 主要是数据预处理文件和模型文件
    :param file_path:
    :return:
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 如果文件不存在，抛出异常
            raise NetworkSecurityException(f"文件不存在 {file_path}")
        # 以二进制模式打开文件
        with open(file_path,"rb") as file_obj:
            # 使用pickle模块加载文件中的对象
            return pickle.load(file_obj)
    except Exception as e:
        # 如果发生异常，抛出NetworkSecurityException异常，并附带异常信息
        raise NetworkSecurityException(e,sys) from e


def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    """

    :param X_train:  训练集特征
    :param y_train:  训练集标签
    :param X_test:  测试集特征
    :param y_test:  测试集标签
    :param models:  模型字典，键为模型名称，值为模型对象
    :param params:  参数字典，键为模型名称，值为参数列表
    :return:  模型评估报告，键为模型名称，值为模型在测试集上的R2得分
    """
    try:
        report= {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            # 使用GridSearchCV进行超参数调优
            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)

            # 自动设置最好的参数
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # 预测训练集和测试集
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(X_train,y_train)
            test_model_score = r2_score(X_test,y_test)
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
