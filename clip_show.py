import cv2
import sys
import os
import time

from open_clip.clip_api import ClipDiscriminator

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


def clip_person(image, clip, person_boxes):
    out_image = image.copy()

    for person_box in person_boxes:
        _, x1, y1, w, h, _ = person_box
        x1, y1, x2, y2 = int(x1),int(y1), int(x1+w), int(y1+h)
        player = image[y1:y2, x1:x2, :]

        shape = player.shape

        if shape[0]>0 and shape[1]>0:
            player_score = clip.forward(player)
            if player_score[0] < 0.7:
                cv2.rectangle(out_image, (x1,y1), (x2,y2), (0,255,0) , 1)
            else:
                cv2.rectangle(out_image, (x1,y1), (x2,y2), (255,0,0) , 1)

    return out_image
    

if __name__ == '__main__':
    clip = ClipDiscriminator(["basketball player", "audience"], device='cuda:0')

    frame_2_person_list = read_track_file('./datas/basketball_dataset_01/NBA-cut-1.track') 
    in_video = './datas/basketball_dataset_01/NBA-cut-1.mp4'
    basename = os.path.basename(in_video)
    reader = cv2.VideoCapture(in_video)

    output_video = os.path.join('./datas/basketball_dataset_01', '{}_{}.mp4'.format('clip', basename))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),30, (1920, 1080))

    more = True
    frame_id = -1
    interval = 10

    tic = time.time()
    while more:
        more, frame = reader.read()
        if frame is not None:
            frame_id += 1
            person_boxes = frame_2_person_list.get(str(frame_id), [])

            out_image = clip_person(frame, clip, person_boxes)
            if frame_id % interval == 0:
                toc = time.time()
                print("== frames speed",interval/(toc-tic))
                tic = time.time()
                print(frame_id)
            writer.write(out_image)


