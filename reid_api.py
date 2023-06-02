
import glob
import os

import cv2
import numpy as np
import onnxruntime

def preprocess(original_image, image_height, image_width):
    #original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img

def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

class PersonFeat:
    def __init__(self, model_path):
        self.height = 256
        self.width = 128
        self.ort_sess = onnxruntime.InferenceSession(model_path,  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.ort_sess.get_inputs()[0].name

    def forward(self, image):
        image = preprocess(image, self.height, self.width)
        feat = self.ort_sess.run(None, {self.input_name: image})[0]
        feat = normalize(feat, axis=1)
        return feat

if __name__ == '__main__':
    image = cv2.imread('datas/00000023.jpg')
    person_feat = PersonFeat('./agw_r50.onnx')
    feat = person_feat.forward(image)
    print(feat.shape)
