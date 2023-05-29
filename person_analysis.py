import os
import sys
import cv2
import insightface

def init_detector():
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    return detector

if __name__ == '__main__':
    detector = init_detector()
    image_path = './datas/00000023.jpg'
    base_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    bboxes, kpss = detector.detect(image)
    print('==bbox ', bboxes)
    #print('==kpss ', kpss)
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0) , 1)
    cv2.imwrite('./output/%s' % base_name, image)
