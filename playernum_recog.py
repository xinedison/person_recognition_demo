import cv2
import sys
import os
import numpy as np
import time
import re
import glob
import insightface

def init_detector():
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    return detector
detector = init_detector()

sys.path.append(os.path.expanduser('~/Codes/PaddleOCR'))
from paddleocr import PaddleOCR
ocr_engine = PaddleOCR()

from clip_api import ClipDiscriminator
clipDiscriminator = ClipDiscriminator(["player wear white shirt", "other"])

def recog_image_number(image, only_detect=False):
    bboxes, _ = detector.detect(image)
    out_image = image.copy()

    player_boxes = []
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        person = image[int(y1):int(y2), int(x1):int(x2), :]
        shape = person.shape
        #print("== player shape", player.shape)
        if shape[0]>0 and shape[1] > 0:
            person_red_prob = clipDiscriminator.forward(person.copy())[0]
            if person_red_prob <= 0.6:
                #print("T shirt color not match")
                continue
            else:
                #print("== crop person in white")
                player_boxes.append([int(x1), int(y1), int(x2), int(y2)]) 
    
    for player_box in player_boxes:
        x1, y1, x2, y2 = player_box
        player = image[y1:y2, x1:x2, :]
        player_num = ''
        #im_show(player)
        shape = player.shape
        #print("== player shape", player.shape)
        if shape[0]>0 and shape[1] > 0:
            #cv2.rectangle(out_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0) , 1)
            #print('== recog number ', player.shape)
            ocr_result = ocr_engine.ocr(player, cls=False)
            #print("== get ocr result ", ocr_result)
            for box_recog_text in ocr_result[0]:
                box, recog_text = box_recog_text
                y_ratio = box[0][1]/shape[0]
                x_ratio = box[0][0]/shape[1]
                if (y_ratio >= 0.2 and y_ratio <= 0.4) and \
                    (x_ratio >= 0.2 and x_ratio <= 0.4): # restrict player number position
                    text = recog_text[0]
                    number = re.findall(r'\d+', text)
                    if len(number) > 0:
                        player_num = number[0]
                        print("== recog player number ", player_num)
                        if len(player_num) <= 2:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(out_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0) , 1)
                            cv2.putText(out_image, player_num, (int(x1-10),int(y1-10)), font, 2, (255,0,0), 3)
                else:
                    pass
                    #print("= ignore ratio ", y_ratio, ',', x_ratio)
    return out_image

if __name__ == '__main__':
    in_video = '/home/avs/Downloads/LNBGvsZJCZ_615.ts'
    basename = os.path.basename(in_video)
    output_video = os.path.join('./datas/recog', '{}_{}'.format('playerNum_white', basename))
    
    reader = cv2.VideoCapture(in_video)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),30, (1920, 1080))
    
    more = True
    frame_id = -1
    interval = 10
    tic = time.time()
    while more:
        more, frame = reader.read()
        if frame is not None:
            frame_id += 1
            searched_out = recog_image_number(frame)
            if frame_id % interval == 0:
                toc = time.time()
                print("== frames speed",interval/(toc-tic))
                tic = time.time()
                print(frame_id)
            writer.write(searched_out)
    reader.release()
    writer.release()
