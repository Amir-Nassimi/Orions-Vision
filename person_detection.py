import numpy as np
from ultralytics import YOLO
from singleton_decorator import singleton
from torch import no_grad as Torch_No_Grad
from torch.cuda import is_available as Cuda_Available


@singleton
class Detection:
    def __init__(self, thresh = 45, model_path='yolov8x.pt'):
        self._model = self.Load_Model(model_path,device='0' if Cuda_Available() else 'cpu')
        self.Warm_Up()
        self.thresh = thresh
        
    def Warm_Up(self):
        print("Warming Up Model!")
        with Torch_No_Grad(): self._model.predict(np.zeros((640,640,3)))
        print("Model Warmed Up!")
    
    @staticmethod
    def Load_Model(pth):
        model = YOLO(f'.models/{pth}')
        print("Model Loaded Successfuly!!")
        return model
    
    def Object_Detection(self, img):
        cordinations = []
        
        with Torch_No_Grad(): results = self._model.predict(img, classes=[1])
        
        if len(results[0]) == 0: pass
        else:    
            for result in results[0]:
                for box in result.boxes:
                    probe = box.conf.item() * 100
                    bounding_boxes = box.xyxy.tolist()[0]

                    if probe > self.thresh:
                        x = int(round(bounding_boxes[0]))
                        y = int(round(bounding_boxes[1]))
                        w = int(round(bounding_boxes[2]))
                        h = int(round(bounding_boxes[3]))

                        cordinations.append((img[y:h, x:w], [x, y, h, w]))


        return cordinations