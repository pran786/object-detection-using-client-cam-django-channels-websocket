
from django.shortcuts import render,redirect
from django.urls import reverse
import base64
from .models import FormDataModel
import numpy as np
#from django.http import StreamingHttpResponse
from django.http import StreamingHttpResponse, HttpResponseRedirect
import requests
from rest_framework.response import Response 
import cv2
import torch
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from rest_framework.decorators import api_view
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from yolov5.detect import *
import json
from django.http import JsonResponse
import csv
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
#from ultralytics.yolo.utils.plotting import Annotator

show_vid=False,  # show results
save_txt=False,  # save results to *.txt
save_conf=False,  # save confidences in --save-txt labels
save_crop=False,  # save cropped prediction boxes
save_vid=False, 
#pred_classes = torch.tensor([0,1,2,3,10]) 
imgsz=(1280, 1280)
#uploaded_file_url = 0
source = 0
data2 = None
frame_keys = []

# the api for the front page (form to recieve inputs from the user for peak detection and object detection)
@api_view(['GET','POST'])
def index(request): 
    global START_X,STOP_X,START_Y,STOP_Y,INPUT_VIDEO,CSV_LIMIT_RECORDS,OUTPUT_CSV,INPUT_VIDEO2
    if request.method=='POST':
        formdata = json.loads(json.dumps(request.data))
        print(formdata)
        
        if "Peak_detection" in formdata.keys(): # to check if the user presses peak detection button
            START_X = int(formdata['start_x'])
            STOP_X = int(formdata['stop_x'])
            START_Y = int(formdata['start_y'])
            STOP_Y = int(formdata['stop_y'])
            INPUT_VIDEO = formdata['peak_video']
            CSV_LIMIT_RECORDS = formdata['csv_records_limit']
            OUTPUT_CSV = formdata['output_folder']
            if INPUT_VIDEO.startswith("rtsp") or INPUT_VIDEO.startswith("http") or INPUT_VIDEO.isdigit() or INPUT_VIDEO.endswith('mp4'):
                return HttpResponseRedirect("/peak_detection")
            else:
                return HttpResponseRedirect("/peakfromweb")

            # form_data_instance = FormDataModel.objects.create(
            #     START_X = int(formdata['start_x']),
            #     STOP_X = int(formdata['stop_x']),
            #     START_Y = int(formdata['start_y']),
            #     STOP_Y = int(formdata['stop_y']),
            #     CSV_LIMIT_RECORDS = int(formdata['csv_records_limit']),
            #     OUTPUT_CSV = formdata['output_folder']
            # )
            
        elif "stream_all" in formdata.keys():  #if peak detection not pressed then object detection task
            START_X = int(formdata['start_x'])
            STOP_X = int(formdata['stop_x'])
            START_Y = int(formdata['start_y'])
            STOP_Y = int(formdata['stop_y'])
            INPUT_VIDEO = formdata['peak_video']
            CSV_LIMIT_RECORDS = formdata['csv_records_limit']
            OUTPUT_CSV = formdata['output_folder']                      
            INPUT_VIDEO2 = formdata['video']
            return HttpResponseRedirect('/stream_all')
        else:
            INPUT_VIDEO2 = formdata['video']
            if INPUT_VIDEO2.startswith("rtsp") or INPUT_VIDEO2.startswith("http") or INPUT_VIDEO2.isdigit() or INPUT_VIDEO2.endswith('mp4'):
                
                return HttpResponseRedirect('/stream_video')
            else:
                return HttpResponseRedirect("/streamfromweb")
            
        


    return render(request, 'index.html')

device = select_device('cpu')

#torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload = True)  # local custom model
model = torch.hub.load('ultralytics/yolov5','yolov5s')

model.conf = 0.20
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
from django.shortcuts import render  
from django.http import HttpResponse  
import time

global json_value
json_value = {}
# #out = cv2.VideoWriter('drone_detection.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (1366,720))

seen = 0

#this function is for stream the object detection video 
@torch.no_grad()
def stream(request):
    global frame_keys
    global seen 
    global data2,mydata,mydata2
    global ret
    if INPUT_VIDEO2.isdigit():
        cap = cv2.VideoCapture(int(INPUT_VIDEO2))
    else:
        cap = cv2.VideoCapture(INPUT_VIDEO2)
    mydata = {}
    while True: 
        mydata2 = []
        ret,frame = cap.read()      
        if ret:
            frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
            print(frame.shape)
           
            seen = seen+1
            if seen%2!=0:
                continue

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
                    mydata = {'x1':int(xyxy[0]), 'y1':int(xyxy[1]), 'x2':int(xyxy[2]), 'y2':int(xyxy[3]),'class': names[c],'score':round(float(conf) * 100,2)}
                    mydata2.append(mydata)

                
            frame = annotator.result()
            #frame = frame*255.0
            #data = im.fromarray(frame[:,:,::-1])
            _,data = cv2.imencode('.jpg',frame)
            data = data.tobytes()
            json_value['frame_no ' + str(seen)] = mydata2

            try:
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n' 
            except RuntimeError:
                continue
        
@api_view(['GET'])
@csrf_exempt

#this is the api for object detection
def video_feed(request):
    try:
        
        return StreamingHttpResponse(stream(request='GET'), content_type = 'multipart/x-mixed-replace; boundary=frame')
    except RuntimeError:
        pass
    

#this api shows the object detection video on the front end (video container)
@api_view(['GET','POST'])
def stream_video(request):
    # if request.method=='POST':
    #     btndata = json.loads(json.dumps(request.data))
    #     btnval = btndata['stop']
    #     print(btnval, 'this is the val we need')
    #     return HttpResponseRedirect('')
    return render(request, 'video_stream.html')

#this api shows the peak detection video on the front end (video container)
@api_view(['GET','POST'])
def stream_peakvideo(request):
    return render(request, 'peak_stream.html')


@api_view(['GET','POST'])
def stream_all(request):
    return render(request, 'stream_all.html')


def streamfromweb(request):
    return render(request, 'streamfromweb.html')

def peakfromweb(request):
    return render(request, 'peakfromweb.html')


#returns the detection in json 
def streamjson(request):
   
    yield json_value
                   

def json_data(request):
       
    return StreamingHttpResponse(streamjson(request))

def Peak_jsondata(request):
    try:
        return JsonResponse(Peak_Json,safe=False)
    except:
        return JsonResponse({},safe=False)


#saves csv file
def save_csv(current_folder_name,times, x_values, y_values,xy_products, output_csv):
    
    df = pd.DataFrame(
        {
            'time': times,
            'x': x_values,
            'y': y_values,
            'formula' : xy_products
        }
    )
    date_saved = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.rename(f"{output_csv}/{current_folder_name}", f"{output_csv}/{date_saved}")
    df.to_csv(f"{output_csv}/{date_saved}/peaks-detections_{date_saved}.csv")


#we are not using this Api
@api_view(['POST'])
def stop_video(request):
    global btnval,apihit
    if request.method=='POST':
        apihit = None
        btndata = json.loads(json.dumps(request.data))
        btnval = btndata['stop']
        if btnval=='true':
            print(btnval, 'this is the val we need')
            return HttpResponseRedirect('/')
        elif btnval=='new':
            return HttpResponseRedirect('/')

#function for streaming the peak detection video
def peak_stream(request):
    global CURRENT_FOLDER_NAME,CSV_NAME,XY_PRODUCT
    global WAITING_RECORDS_LIMIT,Peak_Json
    WAITING_RECORDS_LIMIT=False
    MIN_CONFIDENCE = 0.4
    CSV_NAME = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'+'.csv'
    PEAK_DETECTOR_MODEL_PATH = "best.pt"
    model = YOLO(PEAK_DETECTOR_MODEL_PATH)
    SHOW_BOXES = True
    capture = cv2.VideoCapture(0 if INPUT_VIDEO == "0" else INPUT_VIDEO)
	#capture = cv2.VideoCapture(1)

    times = []
    x_values = []
    y_values = []
    xy_products = []
    Peak_Json = {}
    if not os.path.exists(OUTPUT_CSV):
        os.mkdir(OUTPUT_CSV)

    try:
        while capture.isOpened():

            ret, img = capture.read()
            if not ret:
                break

            
            results = model.predict(img, conf=0.6, classes=[0, 1, 2, 3])
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
                        cv2.imwrite(f"{OUTPUT_CSV}/{CURRENT_FOLDER_NAME}/{date_saved}-{round(capture.get(cv2.CAP_PROP_POS_MSEC))}.jpg", img)

                    if len(times) == int(CSV_LIMIT_RECORDS):
                        save_csv(CURRENT_FOLDER_NAME,times, x_values, y_values,xy_products, OUTPUT_CSV)
                        times, x_values, y_values,xy_products = [], [], [],[]
                        WAITING_RECORDS_LIMIT=False

                _,data = cv2.imencode('.jpg',img)
                data = data.tobytes()
            try:
                
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n' 
            except RuntimeError:
                continue
    except KeyboardInterrupt:
        pass
        
    # except:
    #     print('we got some error')

#api for peak detection video
@api_view(['GET','POST'])
def peak_detection(request):
    try:
        
        return StreamingHttpResponse(peak_stream(request='GET'), content_type = 'multipart/x-mixed-replace; boundary=frame')
    except:
        save_csv(times, x_values, y_values,xy_products, OUTPUT_CSV)



def save_to_csv(OUTPUT_CSV, x,y,xy):
    # Get the current time
    current_time = datetime.now()
    x = int(x)
    y = int(y)
    # Create the folder if it doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        os.makedirs(OUTPUT_CSV)

    # Build the CSV file path
    csv_file_path = os.path.join(OUTPUT_CSV, CSV_NAME)

    file_exists = os.path.exists(csv_file_path)

    
    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(['Time', 'x', 'y','Formula'])

        # Write the data to the CSV file
        csv_writer.writerow([current_time, x, y,xy])


