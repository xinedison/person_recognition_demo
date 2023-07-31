import os
import sys
import cv2
import insightface
import torch
import torchvision.models as models
import torch.onnx
sys.path.append('./detectors/yolo7')
from api import Yolo7


def init_detector():
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
    return detector

def detect_img():
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

def onnx_export():
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True) 
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(resnext50_32x4d, dummy_input, 'resnet50_onnx_model.onnx', verbose=False)
    output = resnext50_32x4d(dummy_input)
    print(output.shape)

if __name__ == '__main__':
    #image_path = './datas/00000023.jpg'
    #frame = cv2.imread(image_path)
    #detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')
    #players = detector.player_det(frame, row=1, col=1)
    #print(players)
    onnx_export()


