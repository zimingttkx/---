"""
这个文件用来启动整个训练管道 以及和前端交互的各项功能实现
"""
import os
import sys

import certifi
from boto3 import client

from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,UploadFile,Request,File
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
import numpy as np


# database = client(DATA_INGESTION_DATABASE_NAME)
# collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origin = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train",tags=["training"])
async def train_model(response=Response("训练管道已经成功开启了")):
    """
    这个接口用来启动整个训练管道
    训练模型
    :return:
    """
    try:
        # 创建训练管道
        train_pipeline = TrainingPipeline()
        # 运行训练管道
        train_pipeline.run_pipeline()

        return response

    except Exception as e:
        # 抛出网络安全异常
        raise NetworkSecurityException(e,sys)


if __name__ == "__main__":
    app_run(app,host="0.0.0.0",port=8000)