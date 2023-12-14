import random
import cv2
import numpy as np
from datetime import datetime
from singleton_decorator import singleton


@singleton
class FrameUtils:
    def __init__(self):
        self.rgb_colors = {}

    @staticmethod
    def merge_frame_dictionaries(*keywords):
        new_frame, counter = {}, 0

        for key in keywords:
            for indx in key:
                if key[indx]['Status'] == '-':
                    continue
                else:
                    new_frame[counter] = key[indx].copy()
                    counter += 1

        return new_frame

    @staticmethod
    def crop_persons(imgs, new_frame):
        indx_ = 0
        for img, cordinations in imgs:
            x = cordinations[0]
            y = cordinations[1]
            h = cordinations[2]
            w = cordinations[3]

            img_ = img.copy()

            if img_.shape[0] < 300:
                continue
            elif img_.shape[1] < 140:
                continue
            else:
                new_frame[indx_] = {
                    'Img': img_,
                    'Status': '-',
                    'Label': 'Unknown',
                    'Coordination': [x, y, w, h],
                    'Last_PID_Run': datetime.now()
                }
                indx_ += 1

        return new_frame

    def write_on_image(self, label, cor, img_original):
        ROI_Region = [0, 0, 0, 0]
        try:
            color = self.rgb_colors[label]
        except KeyError:
            r = random.randrange(1, 256, 20)
            g = random.randrange(1, 256, 20)
            b = random.randrange(1, 256, 20)
            color = (r, g, b)
            self.rgb_colors[label] = color

        img_original = cv2.rectangle(np.array(img_original), (cor[0] + ROI_Region[0], cor[1] + ROI_Region[1], abs(cor[2] - (cor[0])), abs(cor[3] - (cor[1]))), color, 2)
        img_original = cv2.putText(np.array(img_original), f"No: {label}", (cor[0] + ROI_Region[0], cor[1] + ROI_Region[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return img_original