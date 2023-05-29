import numpy
import torch
from PIL import Image
import cv2

import open_clip.src as open_clip


class ClipFeat:
    def __init__(self, device='cuda:0'):
        #self.prompts = prompts
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
                                                                               pretrained='laion400m_e32')
        self.model.to(device)
        self.device = torch.device(device)

    def forward(self, image, decimal=4):
        """
        Zero Shot Classification
        Args:
            image: cv2 / Image / image_path
            decimal: digits number to save

        Returns: Probabilities of each class

        """
        if isinstance(image, numpy.ndarray):
            image = self.preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        elif isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0)
        elif isinstance(image, str):
            image = self.preprocess(Image.open(image)).unsqueeze(0)
        else:
            raise "Unknown Image Type!"

        #text = self.tokenizer(self.prompts)

        image = image.to(self.device)
        #text = text.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
             
            return image_features


if __name__ == '__main__':
    c = ClipFeat()
    img = './open_clip/CLIP.png'
    res = c.forward(img)
    print("clip feat :", res.shape)
