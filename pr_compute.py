import os
import json

def parse_label(label_folder):
    img_2_labels = {}
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)
        img_key = label_file[:-5]
        with open(label_path, 'r') as fin:
            labels = json.loads(fin.read())
            img_players = []
            for person in labels['shapes']:
                box_point = person['points']
                x1, y1 = box_point[0]
                x2, y2 = box_point[1]
                box = [int(x1), int(y1), int(x2), int(y2)]
                player_num = person['group_id']
                img_players.append((box, player_num))
            img_2_labels[img_key] = img_players 
    return img_2_labels

def parse_pred(pred_file):
    img_2_pred_boxes = {}
    with open(pred_file, 'r') as fin:
        for line in fin.readlines():
            items = line.split(',')
            img_name = items[0][:-4]
            pred_boxes = img_2_pred_boxes.get(img_name, [])
            box = list(map(int, items[1:5]))
            player_num = int(items[5])
            pred_boxes.append((box, player_num))
            img_2_pred_boxes[img_name] = pred_boxes
    return img_2_pred_boxes

if __name__ == '__main__':
    label_folder = '/mnt/nas/ActivityDatas/篮球海滨标注/json_labels'
    pred_path = './datas/chouzhen_number.all'
    img_2_pred_boxes = parse_pred(pred_path)
    print("==pred ", img_2_pred_boxes)

    img_2_labels = parse_label(label_folder)
    print("==label ", img_2_labels)
    
    tp = 0
    total_pred = 0
    total_label = 0
    for img_key in img_2_labels:
        label_players = img_2_labels[img_key]
        label_list = []
        for box, label_number in label_players:
            label_list.append(label_number)
        total_label += len(label_list)
        pred_list = []
        pred_players = img_2_pred_boxes.get(img_key, [])
        for box, pred_number in pred_players:
            pred_list.append(pred_number) 
        total_pred += len(pred_list)
        for pred in pred_list:
            if pred in label_list:
                tp += 1
    print("== tp ", tp, " pred ", total_pred, " label ", total_label)
    print("== final precision ", tp/total_pred, " recall ", tp/total_label)
