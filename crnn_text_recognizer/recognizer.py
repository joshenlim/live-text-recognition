import torch
import cv2
from collections import OrderedDict
from torch.autograd import Variable
from crnn_text_recognizer.crnn import CRNN
from crnn_text_recognizer.utils import strLabelConverter
from crnn_text_recognizer.utils import resizeNormalize
from utils.logger import Logger
from PIL import Image

log = Logger()
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
converter = strLabelConverter(alphabet)

'''
Handles the text recognition task in the pipeline for scene text recognition
Using a self-trained CRNN model via the PyTorch implementation from the URL:
https://github.com/meijieru/crnn.pytorch
Current model has accuracy of 89.52%, trained for 8 hours and 27 minutes over 2,000,000 images
Performance on IIIT5k: 75.30%
Performance on ICDAR13: 84.71%
'''

class CRNNRecognizer:
    def __init__(self, model_path):
        log.info('Loading pre-trained CRNN Recognizer model')
        self.model = CRNN(32, 1, 37, 256)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            state_dict_rename[name] = v
        self.model.load_state_dict(state_dict_rename)

    def predict(self, image):
        transformer = resizeNormalize((100, 32))
        image = Image.fromarray(image).convert('L')
        image = transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()

        image = image.view(1, *image.size())
        image = Variable(image)

        self.model.eval()
        preds = self.model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        return sim_pred