####default lib
import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
####need install lib
import uvicorn
import cv2
import traceback
import asyncio
import numpy as np
####custom modules
import rcode
####default lib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from configparser import ConfigParser
####
now = datetime.datetime.now()
#######################################
import torch

from src import util
from src.net import Net
from src.parameters import Parameters
from src.processing_image import warp_image

#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))
MODEL_PATH = str(config.get('model', 'MODEL_PATH'))
#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w",
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
#######################################
####LOAD MODEL HERE
net = Net()
net.load_model(MODEL_PATH)
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("API READY")
#######################################
@app.post('/predict')
async def predict(data: PredictData):
    ###################
    #####
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            images = jsonable_encoder(data.images)
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        for image in images:
            image_decoded = base64.b64decode(image)
            jpg_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
            process_image = cv2.imdecode(jpg_as_np, flags=1)
            process_image = cv2.resize(process_image,(512,256))
            predicts = net.predict(process_image, warp=False)
            image_points = net.get_image_points()
            cv2.imwrite("_tested.jpg", image_points)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_binary')
async def predict_binary(binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        process_image = cv2.resize(process_image,(512,256))
        predicts = net.predict(process_image, warp=False)
        image_points = net.get_image_points()
        # cv2.imwrite("_tested.jpg", image_points)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result
        
@app.post('/predict_multi_binary')
async def predict_binary(binary_files: Optional[List[UploadFile]] = File(None)):
    ###################
    #####
    logger.info("predict_multi_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file_list = []
            for binary_file in binary_files:
                bytes_file_list.append(await binary_file.read())
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        process_image_list = []
        for bytes_file in bytes_file_list:
            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            process_image = cv2.resize(process_image,(512,256))
            predicts = net.predict(process_image, warp=False)
            image_points = net.get_image_points()
            cv2.imwrite("_tested.jpg", image_points)
            process_image_list.append(process_image)

        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_multipart')
async def predict_multipart(argument: Optional[float] = Form(...),
                binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_multipart")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        process_image = cv2.resize(process_image,(512,256))
        predicts = net.predict(process_image, warp=False)
        image_points = net.get_image_points()
        cv2.imwrite("_tested.jpg", image_points)
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)

