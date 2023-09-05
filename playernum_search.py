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

def detect_frame_ocr_rec(image, ocr_engine):
    out_image = image.copy()
    playernum_boxes = []


    number_boxes = detector.detect(image, cls_ids=[0])
    for number_box in number_boxes:
        x1, y1, x2, y2, _, _ = number_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out_image, (x1,y1), (x2,y2), (0,255,0) , 1)

        number_img = out_image[y1:y2, x1:x2, :]
        number_result = ocr_engine.ocr(number_img, det=False, rec=True, cls=False)
        text, score = number_result[0][0]
        print("== recog number ", number_result)
        number = re.findall(r'\d+', text)

        if score >= 0.6 and len(number) > 0:
            player_num = number[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out_image, player_num, (int(x1-10), int(y1-10)), font, 2, (255, 0, 0), 3)
            playernum_boxes.append([x1, y1, x2, y2, player_num])

    return out_image, playernum_boxes

    

def path_2_img_key(path):
    image_name = os.path.basename(path)
    number = os.path.basename(os.path.dirname(path))
    return '{}+{}'.format(number, image_name)


def init_db(feat_extractor, db_path_reg):
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

    
if __name__ == '__main__':
    #cnt_number()
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/number_best_v1.pt')

    in_video = './datas/basketball_dataset_01/CBA-cut11.mp4'
    reader = cv2.VideoCapture(in_video)

    basename = os.path.basename(in_video)
    output_video = os.path.join('./datas/basketball_dataset_01', '{}_{}.mp4'.format('yolo7_num_check', basename))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),30, (1920, 1080))

    output_txt = os.path.join('./datas/basketball_dataset_01', '{}_{}.txt'.format('yolo7_playerNum', basename))
    fout = open(output_txt, 'w')


    #clip_feat = ClipFeat()
    #db_path = './datas/Dataset_jersey_number/train_number/*/*.jpg'
    #searcher = init_db(clip_feat, db_path)

    more = True
    frame_id = -1
    interval = 10

    tic = time.time()
    while more:
        more, frame = reader.read()
        if frame is not None:
            frame_id += 1
            #out_image = detect_frame_number(frame, clip_feat, searcher)

            out_image, playernum_boxes = detect_frame_ocr_rec(frame, ocr_engine)
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
