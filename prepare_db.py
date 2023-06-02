import sys
import os
import json
import cv2
import glob

def parse_label(json_path):
    with open(json_path, 'r') as fin:
        labels = json.loads(fin.read())
        persons = labels['shapes']
    return persons 

if __name__ == '__main__':
    dst_folder = './datas/arsenal_players'
    input_folder = '/home/avs/Codes/PaddleDetection/Arsenal_football_club/10039251_YBS_live55_ikdntk_3min_cut.ts_imgs/'
    imgs = glob.glob(input_folder+'*.jpg')
    for img_path in imgs:
        label_path = img_path[:-3]+'json'
        image = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        if os.path.exists(label_path):
            print('==parse ', label_path)
            labels = parse_label(label_path)
            for person in labels:
                person_box = person['points']
                person_number = person['group_id']
                x1, y1 = person_box[0]
                x1, y1 = int(x1), int(y1)
                x2, y2 = person_box[1]
                x2, y2 = int(x2), int(y2) 
                person_dst_folder = os.path.join(dst_folder, str(person_number))
                if not os.path.exists(person_dst_folder):
                    os.makedirs(person_dst_folder)
                dst_path = os.path.join(person_dst_folder, '{}_{}.jpg'.format(img_name[:-4], person_number))
                person = image[y1:y2, x1:x2]
                print("{}, {}, {}, {}, {}".format(person_number, x1, x2, y1, y2))
                print(person.shape)
                print("==write to", dst_path)
                cv2.imwrite(dst_path, person)
