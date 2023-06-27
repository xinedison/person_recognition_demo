import sys
import os
import cv2
import traceback

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


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

def read_face_file(face_file, threshold=0.57):
    frame_2_face_list = {}
    with open(face_file, 'r') as fin:
        for line in fin.readlines():
            frame_idx, x1, y1, x2, y2, face_id, score = line.split(',')
            if float(score) < threshold:
                continue
            face_list = frame_2_face_list.get(frame_idx, [])
            face_box = tuple(map(float, (face_id, x1, y1, x2, y2)))
            face_list.append(face_box)
            frame_2_face_list[frame_idx] = face_list
    return frame_2_face_list

def read_player_num_file(person_file):
    frame_2_player_num = {}
    with open(person_file, 'r') as fin:
        for line in fin.readlines():
            frame_idx, x1, y1, x2, y2, player_number = line.split(',')

            player_list = frame_2_player_num.get(frame_idx, [])
            player_box = tuple(map(float, (player_number, x1, y1, x2, y2)))
            player_list.append(player_box)
            frame_2_player_num[frame_idx] = player_list
    return frame_2_player_num

def is_rectangle_cross(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    if x1<x2 and y1<y2:
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
    

def is_rectangle_in(rect1, rect2):
    if (rect1[0] > rect2[0]) and (rect1[0] < rect2[2]) and \
        (rect1[1] > rect2[1]) and (rect1[1] < rect2[3]) and \
        (rect1[2] > rect2[0]) and (rect1[2] < rect2[2]) and \
        (rect1[3] > rect2[1]) and (rect1[3] < rect2[3]):
        return True
    return False 

def cnt_face(face_id_list):
    face_id_2_cnt = {}
    for face_id in face_id_list:
        cnt = face_id_2_cnt.get(face_id, 0)
        cnt += 1
        face_id_2_cnt[face_id] = cnt
    sorted_face_id_cnt = sorted(face_id_2_cnt.items(), key=lambda kv:kv[1], reverse=True)
    return sorted_face_id_cnt[0] # get id key

if __name__ == '__main__':
    person_track_file = '/home/avs/Codes/PaddleDetection/output/LNBGvsZJCZ_615.txt'
    face_file = '/home/avs/Codes/face_recognition/datas/recog/faceid_LNBGvsZJCZ_615.ts.txt'
    player_num_file = '/home/avs/Codes/face_recognition/datas/recog/playerNum_white_LNBGvsZJCZ_615.ts.txt'

    frame_2_person_list = read_track_file(person_track_file)

    frame_2_face = read_face_file(face_file)
    frame_2_player_num = read_player_num_file(player_num_file)
    person_2_face = {}
    for frame_idx, person_list in frame_2_person_list.items():
        if frame_idx in frame_2_face:
            face_list = frame_2_face[frame_idx]
            for face in face_list:
                face_id = face[0]
                face_box = face[1:5]
                for person in person_list:
                    person_id = person[0]
                    person_x1, person_y1, person_w, person_h = person[1:5]
                    person_box = [person_x1, person_y1, person_x1+person_w, person_y1+person_h]
                    if is_rectangle_cross(face_box, person_box) and (cross_area(face_box, person_box)>0.95):
                        face_list = person_2_face.get(person_id, [])
                        face_list.append(face_id)
                        person_2_face[person_id] = face_list
                    #person_2_face[person_id] = person_id
    person_2_player_num = {}
    for frame_idx, person_list in frame_2_person_list.items():
        if frame_idx in frame_2_player_num:
            player_num_list = frame_2_player_num[frame_idx]
            for player in player_num_list:
                player_id = player[0]
                player_box = player[1:5]
                for person in person_list:
                    person_id = person[0]
                    person_x1, person_y1, person_w, person_h = person[1:5]
                    person_box = [person_x1, person_y1, person_x1+person_w, person_y1+person_h]
                    if is_rectangle_cross(player_box, person_box):
                        cross_percent = cross_area(player_box, person_box)
                        print("== cross percent ", cross_percent)
                        if cross_percent >= 0.85:
                            player_list = person_2_player_num.get(person_id, [])
                            player_list.append(player_id)
                            person_2_player_num[person_id] = player_list
                
        
    #print(person_2_face)
    video_file = '/home/avs/Downloads/LNBGvsZJCZ_615.ts'
    output_dir = 'output'
    cnt_thresh = 2

    capture = cv2.VideoCapture(video_file)
    video_out_name = os.path.split(video_file)[-1]
    
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    out_path = os.path.join(output_dir, video_out_name)
    video_format = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_id = 0

    while (1):
        ret, frame = capture.read()
        frame_id += 1
        if not ret:
            break

        text_scale = max(0.5, frame.shape[1] / 3000.)
        text_thickness = 2
        line_thickness = max(1, int(frame.shape[1] / 500.))

        try:
            if str(frame_id) not in frame_2_person_list:
                print('== no person in frame ', frame_id)
                continue
            person_list = frame_2_person_list[str(frame_id)]
            for person_box in person_list: 
                person_id, x1, y1, w, h, score = person_box

                intbox = tuple(map(int, (x1, y1, x1+w, y1+h)))
                if person_id in person_2_face: # find person face
                    face_id_list = person_2_face[person_id]
                    face_id_max_cnt = cnt_face(face_id_list)
                    if face_id_max_cnt[1] >= cnt_thresh:
                        face_id = face_id_max_cnt[0]
                        color = get_color(abs(face_id))
                        id_text = 'ID: {}'.format(int(face_id))
 
                        cv2.rectangle(
                            frame, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                        cv2.putText(
                            frame,
                            id_text, (intbox[0], intbox[1] - 25),
                            cv2.FONT_ITALIC,
                            text_scale, (0, 255, 255),
                            thickness=text_thickness)
                if person_id in person_2_player_num:
                    player_id_list = person_2_player_num[person_id]
                    player_id_max_cnt = cnt_face(player_id_list)
                    if player_id_max_cnt[1] >= cnt_thresh:
                        player_id = player_id_max_cnt[0]
                        color = get_color(abs(player_id))
                        id_text = 'PID: {}'.format(player_id)
                        cv2.rectangle(
                            frame, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                        cv2.putText(
                            frame,
                            id_text, (intbox[0], intbox[1] - 25),
                            cv2.FONT_ITALIC,
                            text_scale, (0, 255, 255),
                            thickness=text_thickness)

        except Exception as e:
            print("== ignore frame ", frame_id, e, traceback.print_exc()) 
        writer.write(frame)

    capture.release()
    writer.release()
