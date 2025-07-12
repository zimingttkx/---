# 文件名: app.py
# 这个文件与 index.html 在同一个目录下

import os
import sys
import io
import certifi
from dotenv import load_dotenv
import pandas as pd
import numpy as np  # 确保导入 numpy
import uvicorn

# 加载环境变量
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
ca = certifi.where()

# FastAPI 和相关模块
from fastapi import FastAPI, UploadFile, Request, File, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

# 你的项目模块
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="网络安全威胁预测 API",
    description="一个用于训练模型和进行威胁预测的 API 服务",
    version="1.0.0"
)

# --- 中间件 (Middleware) ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模板配置 ---
# 【关键修改 1】
# 因为 index.html 和 app.py 在同一目录，所以模板目录就是 "." (当前目录)
templates = Jinja2Templates(directory=".")


# --- API 端点 (Endpoints) ---

@app.get("/", tags=["Frontend"], response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    直接从当前目录返回 index.html 页面。
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train", tags=["Training"])
async def train_model():
    """
    启动训练管道。
    """
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response(content="训练管道已成功启动并运行完毕。", status_code=200)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


@app.post("/predict", tags=["Prediction"])
async def predict_from_file(file: UploadFile = File(...)):
    """
    接收上传的 CSV 文件，进行预测并返回 JSON 格式的结果。
    """
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse(status_code=400, content={"message": "文件格式错误，请上传 CSV 文件。"})

        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(buffer)

        # 【请注意】你需要在这里提供一个真实存在的模型路径
        model_path = "path/to/your/trained_model.pkl"  # <--- ！！！请务必修改这里！！！

        if not os.path.exists(model_path):
            logging.error(f"模型文件未找到: {model_path}")
            return JSONResponse(status_code=500, content={"message": "模型文件未找到，请确认路径或先运行训练。"})

        model = load_object(file_path=model_path)

        predictions = model.predict(df)

        df['prediction'] = np.where(predictions == 1, '危险 (Malicious)', '安全 (Benign)')

        response_data = df.to_dict(orient='records')

        return JSONResponse(content=response_data)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --- 自定义异常处理器 ---
@app.exception_handler(NetworkSecurityException)
async def network_security_exception_handler(request: Request, exc: NetworkSecurityException):
    return JSONResponse(
        status_code=500,
        content={"message": f"发生了一个内部错误: {exc.error_message}"}
    )


# --- 启动命令 ---
if __name__ == "__main__":
    # 【关键修改 2】
    # 因为我们的文件名是 app.py，所以这里的字符串是 "app:app"
    # 格式为 "文件名:FastAPI实例名"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)