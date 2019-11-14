import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape): # download, read data 등등을 하는 파트
        hr_height, hr_width = hr_shape # 256 X 256
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose( # 고해상도로 만들 데이터인 low resolution data
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC), # 기본 width, height에서 4를 나눈 값으로 resize
                transforms.ToTensor(), # image값을 tensor형태로 변환
                transforms.Normalize(mean, std), # 위에서 선언한 mean, std값을 사용하여 0~1사이로 Normalize
            ]
        )
        self.hr_transform = transforms.Compose( # high resolution image 데이터
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC), # BICUBIC : 주변 16개의 픽셀을 사용하여 처리
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*")) # 오름차순으로 파일 정렬

    def __getitem__(self, index): # 인덱스에 해당하는 아이템을 넘겨주는 파트
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr} # map 형태로 반환

    def __len__(self): # data size를 넘겨주는 파트
        return len(self.files) # 파일 길이 반환 ( 총 이미지 수 )
