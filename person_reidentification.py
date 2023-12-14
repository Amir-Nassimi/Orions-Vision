import torch
import torchreid.models as models
import torchvision.transforms as transforms

from singleton_decorator import singleton


@singleton
class PersonReidentification:
    def __init__(self, model_path='model.pth.tar-80'):
        self._reidentification_model = None
        self._model_path = f'./models/{model_path}'
        
    @property
    def reidentification_model(self):
        if self._reidentification_model is None:
            self._reidentification_model = self.load_pretrained_model(self._model_path)
        return self._reidentification_model

    # @reidentification_model.setter
    # def reidentification_model(self, new_model):
    #     self._reidentification_model = new_model

    @staticmethod
    def load_pretrained_model(model_path):
        reidentification_model = models.build_model(
            name="xception",
            num_classes=751,
            loss="softmax",
            pretrained=True
        )

        checkpoint = torch.load(model_path)
        reidentification_model.load_state_dict(checkpoint['state_dict'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reidentification_model = reidentification_model.to(device)
        reidentification_model.eval()

        return reidentification_model

    @staticmethod
    def check_similarity(distns, thresh):
        try:
            indx, dist = min(distns, key=lambda x: x[1])
        except ValueError:
            dist = 9e9

        if dist < thresh:
            return indx, dist, True
        else:
            return 'Unknown', None, False
        
    @staticmethod
    def pre_proc(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)
        return img
