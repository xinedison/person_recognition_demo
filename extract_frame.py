import os
import cv2

in_video = '/home/avs/Downloads/LNBGvsZJCZ_615.ts'
output_imgs = './datas/LNBGvsZJCZ_615.ts_imgs'

reader = cv2.VideoCapture(in_video)

frame_id = 1
more = True

while more:
    more, frame = reader.read()
    if frame is not None:
        out_img = os.path.join(output_imgs, '{}.jpg'.format(frame_id))
        cv2.imwrite(out_img, frame)
        frame_id += 1
        if frame_id % 100 == 0:
            print("== extract ", frame_id)
