import cv2
import sys
import os
import glob
from cpu_searcher import numpy_searcher
from reid_api import PersonFeat
import insightface

def init_detector():
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    return detector

def path_2_img_key(path):
    img_name = os.path.basename(path)
    team_name = os.path.basename(os.path.dirname(path))
    return '{}+{}'.format(team_name, img_name)

def read_image_to_data(image_path):
    image_data = cv2.imread(image_path)
    return image_data


def init_db(feat_extractor):
    searcher = numpy_searcher(10, 2048)
    db_pathes = glob.glob("./datas/team_db/*/*.jpg")
    for db_path in db_pathes:
        img_key = path_2_img_key(db_path)
        img_data = read_image_to_data(db_path)
        feat = feat_extractor.forward(img_data)[0].tolist()
        searcher.update(img_key, feat)
    return searcher

def crop_db(detector):
    #db_img_path = '/mnt/nas/ActivityDatas/face_recog_datas/LNBGvsZJCZ_615.ts_imgs/7911.jpg'
    db_img_path = '/home/avs/Codes/face_recognition/datas/basketball_dataset_01/CBA-cut11.mp4_imgs/00000026.jpg'
    db_dst_folder = './datas/team_db/CBA-cut11'
    image = cv2.imread(db_img_path)
    bboxes, _  = detector.detect(image)
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2, _ = bbox
        person = image[int(y1):int(y2), int(x1):int(x2), :]
        shape = person.shape
        if shape[0]>0 and shape[1]>0:
            dst_path = os.path.join(db_dst_folder, '{}.jpg'.format(idx))
            cv2.imwrite(dst_path, person)

def search_image(image, detector, feat_extractor, searcher):
    bboxes, _ = detector.detect(image)
    out_image = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        person = image[int(y1):int(y2), int(x1):int(x2), :]
        shape = person.shape
        if shape[0]>0 and shape[1]>0:
            cv2.rectangle(out_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0) , 2) 
            test_feat = feat_extractor.forward(person)
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
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out_image, team_name, (int(x1-10),int(y1-10)), font, 1, (255,0,0), 3)
    return out_image

if __name__ == '__main__':
    detector = init_detector()
    crop_db(detector)
    '''
    person_feat = PersonFeat('./agw_r50.onnx')
    searcher = init_db(person_feat)
    
    img_folder = '/mnt/nas/ActivityDatas/face_recog_datas/LNBGvsZJCZ_615.ts_imgs/'
    dst_folder = './datas/team_out/'
    for idx, image_name in enumerate(os.listdir(img_folder)):
        if idx % 10 == 0:
            print('== handle ', idx)
        image_path = os.path.join(img_folder, image_name)
        image = cv2.imread(image_path)
        out_image = search_image(image, detector, person_feat, searcher)
        cv2.imwrite(os.path.join(dst_folder, image_name), out_image)
    ''' 
