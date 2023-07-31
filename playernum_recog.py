import cv2
import sys
import os
import numpy as np
import time
import re
import glob
import insightface

from cpu_searcher import numpy_searcher
from reid_api import PersonFeat

def init_detector():
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    return detector
#detector = init_detector()

sys.path.append(os.path.expanduser('~/Codes/PaddleOCR'))
from paddleocr import PaddleOCR
ocr_engine = PaddleOCR()

#from clip_api import ClipDiscriminator
#clipDiscriminator = ClipDiscriminator(["player wear white shirt", "other"])


def recog_image_number(image, only_detect=False):
    bboxes, _ = detector.detect(image)
    
    out_image = image.copy()

    player_boxes = []
    playernum_boxes = []
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        person = image[int(y1):int(y2), int(x1):int(x2), :]
        shape = person.shape
        #print("== player shape", player.shape)
        if shape[0]>0 and shape[1] > 0:
            #person_red_prob = clipDiscriminator.forward(person.copy())[0]
            #if person_red_prob <= 0.6:
            #    #print("T shirt color not match")
            #    continue
            #else:
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
                if (y_ratio >= 0.0 and y_ratio <= 0.5) and \
                    (x_ratio >= 0.0 and x_ratio <= 0.5): # restrict player number position
                    text = recog_text[0]
                    number = re.findall(r'\d+', text)
                    if len(number) > 0:
                        player_num = number[0]
                        print("== recog player number ", player_num)
                        if len(player_num) <= 2:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(out_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0) , 1)
                            cv2.putText(out_image, player_num, (int(x1-10),int(y1-10)), font, 2, (255,0,0), 3)
                            playernum_boxes.append([x1, y1, x2, y2, player_num])
                    #else:
                    #    print("== not a number ", text)
                else:
                    pass
                    #print("= ignore ratio ", y_ratio, ',', x_ratio, recog_text)
    return out_image, playernum_boxes


def recog_track_frame(image, person_boxes, feat_extractor, searcher):
    playernum_boxes = []
    out_image = image.copy()
    for person_box in person_boxes:
        _, x1, y1, w, h, _ = person_box
        x1, y1, x2, y2 = int(x1),int(y1), int(x1+w), int(y1+h)
        player = image[y1:y2, x1:x2, :]
        player_num = ''
        shape = player.shape
        if shape[0]>0 and shape[1]>0:
            test_feat = feat_extractor.forward(player)
            topk_keys = searcher.topk(test_feat, topk=3)
            #print(topk_keys)
            team_cnt = {}
            for key, score in topk_keys[0]:
                search_name = key.split('+')[0]
                cnt = team_cnt.get(search_name, 0)
                cnt += 1
                team_cnt[search_name] = cnt
            sort_key = sorted(team_cnt.items(), key=lambda kv:kv[1], reverse=True)
            team_name = sort_key[0][0]

            ocr_result = ocr_engine.ocr(player, cls=False)
            #print("== ocr result ", ocr_result)
            for box_recog_text in ocr_result[0]:
                box, recog_text = box_recog_text
                y_ratio = box[0][1]/shape[0]
                x_ratio = box[0][0]/shape[1]
                if (y_ratio >= 0.1 and y_ratio <= 0.5) and \
                    (x_ratio >= 0.1 and x_ratio <= 0.5): # restrict player number position
                    text = recog_text[0]
                    number = re.findall(r'\d+', text)
                    if len(number) > 0:
                        player_num = number[0]
                        print("== recog player number ", player_num)
                        if len(player_num) <= 2:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            player_num = '{}_{}_N_0'.format(player_num, team_name)
                            cv2.rectangle(out_image, (x1,y1), (x2,y2), (0,255,0) , 1)
                            cv2.putText(out_image, player_num, (x1-10,y1-10), font, 2, (255,0,0), 3)
                            playernum_boxes.append([x1, y1, x2, y2, player_num])
                else:
                    print("== ignore ratio", x_ratio, ' ', y_ratio, ' ', recog_text)
    return out_image, playernum_boxes

def path_2_img_key(path):
    img_name = os.path.basename(path)
    team_name = os.path.basename(os.path.dirname(path))
    return '{}+{}'.format(team_name, img_name)


def init_db(feat_extractor, db_path):
    searcher = numpy_searcher(10, 2048)
    db_pathes = glob.glob(db_path)
    for db_path in db_pathes:
        img_key = path_2_img_key(db_path)
        img_data = cv2.imread(db_path)
        feat = feat_extractor.forward(img_data)[0].tolist()
        searcher.update(img_key, feat)
    return searcher
    

def read_track_file(track_file):
    frame_2_person_list = {}
    with open(track_file, 'r') as fin:
        for line in fin.readlines():
            frame_idx, person_id, x, y, w, h, score, _, _, _ = line.split(",")
            person_list = frame_2_person_list.get(frame_idx, [])
            person_box = tuple(map(float, (person_id, x, y, w, h, score)))
            person_list.append(person_box)
            frame_2_person_list[frame_idx] = person_list
    return frame_2_person_list


def recog_video():
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

def recog_folder():
    input_folder = '/mnt/nas/ActivityDatas/篮球海滨标注/chouzhen_sample'
    output_folder = './datas/chouzhen_number'
    number_outpath = './datas/chouzhen_number.txt'
    
    fout = open(number_outpath, 'w')
    for idx, img_name in enumerate(os.listdir(input_folder)):
        if idx % 10 == 0:
            print("== handle ", idx)
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        searched_out, playernum_boxes = recog_image_number(frame)
        print("== search player number ", img_name)
        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, searched_out)
        
        for player_box in playernum_boxes:
            player_out = ','.join(map(str, player_box))
            line = '{},{}\n'.format(img_name, player_out)
            fout.write(line)


if __name__ == '__main__':
    person_feat = PersonFeat('./agw_r50.onnx')
    db_path = "./datas/team_db/CBA-cut11/*/*.jpg"
    team_searcher = init_db(person_feat, db_path)

    frame_2_person_list = read_track_file('./datas/basketball_dataset_01/CBA-cut11.track') 
    in_video = './datas/basketball_dataset_01/CBA-cut11.mp4'
    basename = os.path.basename(in_video)
    reader = cv2.VideoCapture(in_video)

    output_video = os.path.join('./datas/basketball_dataset_01', '{}_{}.mp4'.format('playerNum_', basename))
    output_txt = os.path.join('./datas/basketball_dataset_01', '{}_{}.txt'.format('playerNum_', basename))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),30, (1920, 1080))
    fout = open(output_txt, 'w')

    more = True
    frame_id = -1
    interval = 10

    tic = time.time()
    while more:
        more, frame = reader.read()
        if frame is not None:
            frame_id += 1
            person_boxes = frame_2_person_list.get(str(frame_id), [])
            out_image, playernum_boxes = recog_track_frame(frame, person_boxes, person_feat, team_searcher)
            for player_box in playernum_boxes:
                player_out = ','.join(map(str, player_box))            
                line = '{},{}\n'.format(frame_id, player_out)
                fout.write(line)

            if frame_id % interval == 0:
                toc = time.time()
                print("== frames speed",interval/(toc-tic))
                tic = time.time()
                print(frame_id)
            writer.write(out_image)

