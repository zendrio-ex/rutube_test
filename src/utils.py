from torchvision import transforms
from config import settings
import torch
import torch.nn.functional as F

max_height, max_width = settings['max_height'], settings['max_width']


class Preparation:
    class MinMaxScaler():
        def __call__(self, image):
            return image.long()/256

    class Cropper():
        def __call__(self, image):
            if (image.size()[1] > max_height) and (image.size()[2] > max_width):
                cropper = transforms.CenterCrop(size=(max_height,
                                                      max_width))
            elif (image.size()[1] > max_height):
                cropper = transforms.CenterCrop(size=(max_height,
                                                      image.size()[2]))
            elif (image.size()[2] > max_width):
                cropper = transforms.CenterCrop(size=(image.size()[1],
                                                      max_width))
            else:
                return image

            return cropper(image)

    class GrayPreprocessing():
        def __call__(self, image):
            if image.size()[0] == 1:
                image = torch.cat((image, image, image), 0)
            return image

    class Padding():
        def __init__(self, max_height: int, max_width: int, fill: int = 0):
            self.fill = fill
            self.max_height = max_height
            self.max_width = max_width

        def __call__(self, image):
            im_size = image.size()
            h_padding = (max_height - im_size[1]) / 2
            v_padding = (max_width - im_size[2]) / 2

            l_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
            t_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
            r_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
            b_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5

            return F.pad(image, (int(l_pad), int(r_pad),
                                 int(t_pad), int(b_pad)))

    def __init__(self):
        self.image_transform = transforms.Compose([
                self.GrayPreprocessing(),
                self.Cropper(),
                self.MinMaxScaler(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
                self.Padding(max_height, max_width, fill=0)
            ])

    def pipeline(self, image):
        image = self.image_transform(image)
        return image


preparation = Preparation()
