import sys
import os
import glob
import cv2

train_dir = './datas/LNBGvsZJCZ_615.ts_imgs'
dst_dir = './datas/train_LNB'

train_files = glob.glob('{}/*.jpg'.format(train_dir))

for i, train_file in enumerate(train_files):
    if i % 20 == 0:
        file_name = os.path.basename(train_file)
        dst_file = os.path.join(dst_dir, file_name)
        im = cv2.imread(train_file)
        im = im[150:950, :, :]
        cv2.imwrite(dst_file, im)
        #shutil.copy(train_file, dst_file)
        


