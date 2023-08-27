import cv2
import os
import pandas as pd
import argparse
from datetime import datetime
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


def save_csv(current_folder_name,times, x_values, y_values, output_csv):
    
    df = pd.DataFrame(
        {
            'time': times,
            'x': x_values,
            'y': y_values
        }
    )
    date_saved = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.rename(f"{output_csv}/{current_folder_name}", f"{output_csv}/{date_saved}")
    df.to_csv(f"{output_csv}/{date_saved}/peaks-detections_{date_saved}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x1', '--start_x')
    parser.add_argument('-x2', '--stop_x')
    parser.add_argument('-y1', '--start_y')
    parser.add_argument('-y2', '--stop_y')
    parser.add_argument('-c', '--confidence', default=0.4)
    parser.add_argument('-v', '--video')
    parser.add_argument('-rl', '--csv_records_limit')
    parser.add_argument('-o', '--output_folder')
    parser.add_argument('-m', '--model_path',
                        default='runs/detect/train/weights/best.pt')
    parser.add_argument('-s', '--show', action="store_true",
                        help="Enable show results")

    args = parser.parse_args()

    MIN_CONFIDENCE = float(args.confidence)
    START_X, STOP_X = int(args.start_x), int(args.stop_x)
    START_Y, STOP_Y = int(args.start_y), int(args.stop_y)
    SHOW_BOXES = bool(args.show)
    INPUT_VIDEO = args.video
    CSV_LIMIT_RECORDS = args.csv_records_limit
    PEAK_DETECTOR_MODEL_PATH = args.model_path
    OUTPUT_CSV = args.output_folder
    global CURRENT_FOLDER_NAME
    global WAITING_RECORDS_LIMIT
    WAITING_RECORDS_LIMIT=False
    
    model = YOLO(PEAK_DETECTOR_MODEL_PATH)
    capture = cv2.VideoCapture(0 if INPUT_VIDEO == "0" else INPUT_VIDEO)
	#capture = cv2.VideoCapture(1)
    times = []
    x_values = []
    y_values = []

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
                        CURRENT_FOLDER_NAME= round(capture.get(cv2.CAP_PROP_POS_MSEC))                        
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

                    times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
                    x_values.append(int(peak_x_normalized))
                    y_values.append(int(peak_y_normalized))

                    if SHOW_BOXES:
                        annotator.box_label(
                            peak_xyxym, f"x {int(peak_x_normalized)} y {int(peak_y_normalized)}")
                        date_saved = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                        cv2.imwrite(f"{OUTPUT_CSV}/{CURRENT_FOLDER_NAME}/{date_saved}-{round(capture.get(cv2.CAP_PROP_POS_MSEC))}.jpg", img)

                    if len(times) == int(CSV_LIMIT_RECORDS):
                        save_csv(CURRENT_FOLDER_NAME,times, x_values, y_values, OUTPUT_CSV)
                        times, x_values, y_values = [], [], []
                        WAITING_RECORDS_LIMIT=False

            if SHOW_BOXES:
                cv2.imshow("Detections", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # by press 'q' you quit the process                    
                    cv2.destroyAllWindows()
                    save_csv(CURRENT_FOLDER_NAME,times, x_values, y_values, OUTPUT_CSV)
                    break


    except KeyboardInterrupt:
        save_csv(times, x_values, y_values, OUTPUT_CSV)
