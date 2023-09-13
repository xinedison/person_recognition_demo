# -*- coding: UTF-8 -*-
import os
import sys
import shutil

if __name__ == '__main__':
    label_folder = '/mnt/nas/ActivityDatas/篮球海滨标注/json_labels'
    img_folder = '/mnt/nas/ActivityDatas/篮球海滨标注/chouzhenPNG_origin'
    dst_folder = '/mnt/nas/ActivityDatas/篮球海滨标注/chouzhenPNG'
    for label_path in os.listdir(label_folder):
        img_name = label_path[:-4]+'png'
        src_path = os.path.join(img_folder, img_name) 
        dst_path = os.path.join(dst_folder, img_name)
        shutil.copy(src_path, dst_path)
