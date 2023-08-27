# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import base64
import cv2
import numpy as np
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
import torch
import csv
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from .views import model,save_to_csv,save_csv
from urllib.parse import parse_qs
from channels.db import database_sync_to_async
from Cards.models import FormDataModel
import os
global START_X,STOP_X,START_Y,STOP_Y,INPUT_VIDEO,CSV_LIMIT_RECORDS,OUTPUT_CSV,INPUT_VIDEO2
model.conf = 0.20
stride, names, pt = model.stride, model.names, model.pt
START_X = 5625
STOP_X = 5925
START_Y = -110
STOP_Y = -19
CSV_LIMIT_RECORDS = 50
OUTPUT_CSV = "outputs"
times = []
x_values = []
y_values = []
xy_products = []
Peak_Json = {}
class FrameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # self.form_data_id = self.scope['url_route']['kwargs']['form_data_id']
        # redirect_url = self.scope['query_string'].decode('utf-8')
        # print(redirect_url, 'this is query string')
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frameData']
            print(len(frame_data))

            # Decode the base64 frame.
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Process the frame (modify this part based on your processing logic).
            frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
            

            results = model(frame, augment= True)

            det = results.pred[0]
            annotator = Annotator(frame, line_width=2,pil= not ascii)
            if det is not None and len(det):
                mylist = []
                for *xyxy, conf, cls in reversed(det):  
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    mylist.append(names[c])
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    #mydata = {'x1':int(xyxy[0]), 'y1':int(xyxy[1]), 'x2':int(xyxy[2]), 'y2':int(xyxy[3]),'class': names[c],'score':round(float(conf) * 100,2)}
                    #mydata2.append(mydata)

                
            frame = annotator.result()
            frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            processed_frame = frame
            
            # Encode the processed frame as base64.
            _, buffer = cv2.imencode('.jpg', processed_frame)
            base64_processed_frame = base64.b64encode(buffer).decode('utf-8')

            # Send the processed frame back to the client through WebSocket
            await self.send(text_data=json.dumps({'processedFrame': base64_processed_frame}))
        except:
            print("error message")




class PeakFrameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # self.form_data_id = self.scope['url_route']['kwargs']['form_data_id']
        global CURRENT_FOLDER_NAME,CSV_NAME,XY_PRODUCT,MIN_CONFIDENCE,SHOW_BOXES
        global WAITING_RECORDS_LIMIT,Peak_Json,times,x_values ,y_values,xy_products,Peak_Json   
        WAITING_RECORDS_LIMIT=False
        MIN_CONFIDENCE = 0.4
        CSV_NAME = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'+'.csv'
        PEAK_DETECTOR_MODEL_PATH = "best.pt"
        self.model2 = YOLO(PEAK_DETECTOR_MODEL_PATH)
        SHOW_BOXES = True
        if not os.path.exists(OUTPUT_CSV):
            os.mkdir(OUTPUT_CSV)
        # redirect_url = self.scope['query_string'].decode('utf-8')
        # print(redirect_url, 'this is query string')
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frameData']
            print(len(frame_data))

            # Decode the base64 frame.
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Process the frame (modify this part based on your processing logic).
            img = cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
            
            results = self.model2.predict(img, conf=0.6, classes=[0, 1, 2, 3])
            annotator = Annotator(img)

            for result in results:
                peak_xyxym, peak_conf = None, 0
                line_xyxy, line_conf = None, 0
                yaxis_xyxy, yaxis_conf = None, 0

                for box in result.boxes:
                    box_xyxy = box.xyxy[0]
                    box_confidence = box.conf
                    class_name = result.names[int(box.cls)]

                    if class_name == "line" and box_confidence > line_conf:
                        line_xyxy, line_conf = box_xyxy, box_confidence
                    if class_name == "peak" and box_confidence > peak_conf:
                        peak_xyxym, peak_conf = box_xyxy, box_confidence
                    if class_name == "yaxis" and box_confidence > yaxis_conf:
                        yaxis_xyxy, yaxis_conf = box_xyxy, box_confidence

                if all(confidence >= MIN_CONFIDENCE for confidence in [peak_conf, line_conf, yaxis_conf]):
                    if not WAITING_RECORDS_LIMIT:                        
                        CURRENT_FOLDER_NAME= datetime.now().strftime("%Y-%m-%d %H-%M-%S")                        
                        if not os.path.exists(f'{OUTPUT_CSV}/{CURRENT_FOLDER_NAME}'):
                            os.mkdir(f'{OUTPUT_CSV}/{CURRENT_FOLDER_NAME}')
                        WAITING_RECORDS_LIMIT=True

                    line_x1, line_x2 = line_xyxy[0], line_xyxy[2]
                    peak_x1, peak_x2 = peak_xyxym[0], peak_xyxym[2]
                    peak_x = ((peak_x1 + peak_x2) / 2)  # Gets middle
                    peak_x_normalized = START_X + \
                        ((peak_x - line_x1) / (line_x2 - line_x1) * (STOP_X - START_X))

                    yaxis_y1, yaxis_y2 = yaxis_xyxy[1], yaxis_xyxy[3]
                    peak_y1 = peak_xyxym[1]
                    peak_y_normalized = - \
                        ((peak_y1 - yaxis_y1) / (yaxis_y2 - yaxis_y1)
                        * (STOP_Y - START_Y)) + STOP_Y

                    times.append(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                    x_values.append(int(peak_x_normalized))
                    y_values.append(int(peak_y_normalized))
                    XY_PRODUCT = int(peak_x_normalized)*int(peak_y_normalized)
                    xy_products.append(XY_PRODUCT)
                    Peak_Json["Time"] = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    Peak_Json["formula"] = XY_PRODUCT
                    save_to_csv(OUTPUT_CSV,peak_x_normalized,peak_y_normalized,XY_PRODUCT)
                    if SHOW_BOXES:
                        annotator.box_label(
                            peak_xyxym, f"x {int(peak_x_normalized)} y {int(peak_y_normalized)}")
                        date_saved = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                        cv2.imwrite(f"{OUTPUT_CSV}/{CURRENT_FOLDER_NAME}/{date_saved}-{datetime.now()}.jpg", img)

                    if len(times) == int(CSV_LIMIT_RECORDS):
                        save_csv(CURRENT_FOLDER_NAME,times, x_values, y_values,xy_products, OUTPUT_CSV)
                        times, x_values, y_values,xy_products = [], [], [],[]
                        WAITING_RECORDS_LIMIT=False




            img = cv2.resize(img,(1280,720),interpolation=cv2.INTER_AREA)
            processed_frame = img
            
            
            # Encode the processed frame as base64.
            _, buffer = cv2.imencode('.jpg', processed_frame)
            base64_processed_frame = base64.b64encode(buffer).decode('utf-8')

            # Send the processed frame back to the client through WebSocket
            await self.send(text_data=json.dumps({'processedFrame': base64_processed_frame}))
        except:
            print("error message")
