import sys 
import os 
import cv2
import time
import glob
import re

sys.path.append('./detectors/yolo7')
from api import Yolo7
from cpu_searcher import numpy_searcher
from clip_api import ClipFeat

from reid_api import PersonFeat


sys.path.append(os.path.expanduser('~/Codes/PaddleOCR'))
from paddleocr import PaddleOCR
ocr_engine = PaddleOCR()


def cnt_number():
    number_root = '/home/avs/Codes/face_recognition/datas/Dataset_jersey_number/train_number'

    total_cnt = 0
    for number_folder in os.listdir(number_root):
        number_path = os.path.join(number_root, number_folder)
        number_cnt = len(os.listdir(number_path))
        total_cnt += number_cnt
        print("{} has {} images".format(number_folder, number_cnt))
    print("== total images ", total_cnt)

def detect_frame_number(image, feat_extractor, searcher):
    out_image = image.copy()

    number_boxes = detector.detect(image, cls_ids=[0])
    for number_box in number_boxes:
        x1, y1, x2, y2, _, _ = number_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        number_img = out_image[y1:y2, x1:x2, :]
        number_feats = feat_extractor.forward(number_img).cpu().numpy()
        topk_keys = searcher.topk(number_feats, topk=1)
        searched_key, score = topk_keys[0][0]
        cv2.rectangle(out_image, (x1,y1), (x2,y2), (0,255,0) , 1)
        print("== searched number ", searched_key, " score ", score)
        if score >= 0.65:
            playernum = searched_key.split('+')[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out_image, playernum, (int(x1-10), int(y1-10)), font, 2, (255, 0, 0), 3)
    return out_image


def get_team_name(feat_extractor, player, searcher):
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
    return team_name


def is_rectangle_cross(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    if x1<=x2 and y1<=y2:
        return True
    else:
        return False

def cross_area(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    cross = (x2-x1) * (y2-y1)
    rect1_area = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])
    percent = cross/rect1_area
    return percent
 


def detect_frame_ocr_rec(image, ocr_engine, person_boxes, feat_extractor, team_searcher, valid_number_list):
    out_image = image.copy()
    playernum_boxes = []

    number_boxes = detector.detect(image, cls_ids=[0])
    for number_box in number_boxes:
        num_x1, num_y1, num_x2, num_y2, _, _ = number_box
        num_x1, num_y1, num_x2, num_y2 = int(num_x1), int(num_y1), int(num_x2), int(num_y2)

        cv2.rectangle(out_image, (num_x1,num_y1), (num_x2,num_y2), (0,255,0) , 1)
        number_img = out_image[num_y1:num_y2, num_x1:num_x2, :]
        number_result = ocr_engine.ocr(number_img, det=False, rec=True, cls=False)
        text, score = number_result[0][0]
        #print("== recog number ", number_result)
        number = re.findall(r'\d+', text)
    
        if score >= 0.6 and len(number) > 0:
            player_num_ocr = number[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out_image, player_num_ocr, (int(num_x1-10), int(num_y1-10)), font, 2, (255, 0, 0), 3)
            #playernum_boxes.append([num_x1, num_y1, num_x2, num_y2, player_num_ocr])

            for person_box in person_boxes:
                _, person_x1, person_y1, person_w, person_h, _ = person_box
                person_x1, person_y1, person_x2, person_y2 = int(person_x1),int(person_y1), int(person_x1+person_w), int(person_y1+person_h)

                number_box = [num_x1, num_y1, num_x2, num_y2]
                person_box_rect = [person_x1, person_y1, person_x2, person_y2]
                if is_rectangle_cross(number_box, person_box_rect) and (cross_area(number_box, person_box_rect) >= 0.85):
                    player = image[person_y1:person_y2, person_x1:person_x2, :]
                    shape = player.shape
                    if shape[0]>0 and shape[1]>0:
                        team_name = get_team_name(feat_extractor, player, team_searcher)
                        player_num_team = '{}_{}_N_0'.format(player_num_ocr, team_name)
                        if player_num_team in valid_number_list:
                            playernum_boxes.append([num_x1, num_y1, num_x2, num_y2, player_num_team])
                            break
                        else:
                            print("== ignore invalid number player_num_team ", player_num_team)
       
    return out_image, playernum_boxes

    

def path_2_img_key(path):
    image_name = os.path.basename(path)
    number = os.path.basename(os.path.dirname(path))
    return '{}+{}'.format(number, image_name)

def init_team_db(feat_extractor, db_path):
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


def init_num_db(feat_extractor, db_path_reg):
    searcher = numpy_searcher(510, 512)
    db_pathes = glob.glob(db_path_reg)
    for idx, db_path in enumerate(db_pathes):
        if idx % 100 == 0:
            print("== update ", idx, " to db")
        img_key = path_2_img_key(db_path)
        img_data = cv2.imread(db_path)
        feat = feat_extractor.forward(img_data).cpu().numpy()[0].tolist()
        searcher.update(img_key, feat)
    return searcher

def read_valid_number(number_path):
    valid_numbers = []
    with open(number_path, 'r') as fin:
        for line in fin.readlines():
            valid_numbers.append(line.strip())
    return valid_numbers

    
if __name__ == '__main__':
    #cnt_number()
    person_feat = PersonFeat('./agw_r50.onnx')
    db_path = "./datas/team_db/CBA-cut11/*/*.jpg"
    team_searcher = init_team_db(person_feat, db_path)

    frame_2_person_list = read_track_file('./datas/basketball_dataset_01/CBA-cut11.track') 

    valid_number_list = read_valid_number('./datas/team_db/CBA_valid_number.txt')

    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/number_best_v1.pt')

    in_video = './datas/basketball_dataset_01/CBA-cut11.mp4'
    reader = cv2.VideoCapture(in_video)

    basename = os.path.basename(in_video)
    output_video = os.path.join('./datas/basketball_dataset_01', '{}_{}.mp4'.format('yolo7_num_team', basename))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),30, (1920, 1080))

    output_txt = os.path.join('./datas/basketball_dataset_01', '{}_{}.txt'.format('yolo7_team_Num', basename))
    fout = open(output_txt, 'w')


    #clip_feat = ClipFeat()
    #db_path = './datas/Dataset_jersey_number/train_number/*/*.jpg'
    #searcher = init_num_db(clip_feat, db_path)

    more = True
    frame_id = -1
    interval = 10

    tic = time.time()
    while more:
        more, frame = reader.read()
        if frame is not None:
            frame_id += 1
            #out_image = detect_frame_number(frame, clip_feat, searcher)
            person_boxes = frame_2_person_list.get(str(frame_id), [])

            out_image, playernum_boxes = detect_frame_ocr_rec(frame, ocr_engine, person_boxes, person_feat, team_searcher, valid_number_list)
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
