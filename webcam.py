from PIL import Image
import cv2
import math 
import functions.utils_rotate as utils_rotate
import os
import time
import argparse
import functions.helper as helper
from ultralytics import YOLO

# load model
yolo_LP_detect = YOLO('models/lp.pt')
yolo_license_plate = YOLO('models/ocr.pt')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture("test_image/test_6.mp4")

# # Get video properties
# frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(vid.get(cv2.CAP_PROP_FPS))

# # Define the codec and create VideoWriter object
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while(True):
    ret, frame = vid.read()
    if not ret:
        break
    
    results = yolo_LP_detect(frame)
    list_plates = results[0].boxes.xyxy.cpu().numpy()
    list_read_plates = set()
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    break
            if flag == 1:
                break
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Write the frame into the file
    # out.write(frame)

vid.release()
# out.release()
cv2.destroyAllWindows()