
BATCH_SIZE = 4
WORKERS = 8
# WORKERS = 0
IMG_SIZE = 512
# IMG_SIZE = 320
# GENDER_SENSITIVE = True
DS_MEAN = 0.1826
DS_STD = 0.1647

mu =  136.72353790613718
sigma =  62.34640414043511


import csv
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torchvision import transforms
from torchvision.transforms import InterpolationMode
transforms.ToTensor()

# from skimage.exposure import equalize_adapthist

import PIL
from PIL import ImageOps, ImageEnhance
from PIL import Image
import albumentations as A
as_tensor = transforms.ToTensor()


class BoneAgeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, male=None, apply_transform=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            male (boolean): If image is from a man
            apply_transform (boolean): Enable random transformations (augmentation)
        """
        self.df_images = pd.read_csv(csv_file, usecols=['fileName', 'male'])
        self.df_labels = pd.read_csv(csv_file, usecols=['boneage', 'male'])
        # self.df_gender = pd.read_csv(csv_file,usecols=['male'])
        
        self.root_dir = root_dir
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(30, translate=(0.1, 0.1), scale=(0.7, 1), shear=None, resample=False, fillcolor=0),
            # transforms.Grayscale(1),
        ]) if apply_transform else None

        # self.transform = A.Compose([
        #     # A.RandomCrop(width=256, height=256),
        #     A.HorizontalFlip(p=0.5),
        #     # A.RandomBrightnessContrast(p=0.2),
        #     A.OneOf([
        #         # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
        #         A.GaussNoise(),    # 将高斯噪声应用于输入图像。
        # ], p=0.2), 
            
        #     # Generate a square region in the image
        #     A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),

        #     # Random images have a great impact on the recognition effect here
        #     # A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.2),

        #     # Randomly shuffle the image in patches
        #     A.RandomGridShuffle(grid=(8,8),p=1,always_apply=False),

        #     A.OneOf([
        #         # A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
        #         A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
        #         A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        # ], p=0.2),

        #     # as_tensor(),
        #     A.ShiftScaleRotate(rotate_limit=45,p=0.4,)
        # ]) if apply_transform else A.Compose([])
        
        if male is not None:
            # 'male' == male
            self.df_images = self.df_images[self.df_images['male'] == male]
            self.df_labels = self.df_labels[self.df_labels['male'] == male]


        # self.df_images = self.df_images.drop('male', axis=1)
        # self.df_labels = self.df_labels.drop('male', axis=1)


    def __len__(self):
        return len(self.df_images)

    def __getitem__(self, idx):        
        # read img file
        # img_path = os.path.join(self.root_dir, self.df_images.iloc[idx, 0])

        # img_path = self.df_images.iloc[idx,0]
        img_path = self.df_images.iloc[idx,1]

        gender_tag = self.df_images.iloc[idx,0]
        if str(gender_tag) == 'True':
            gender = 1.
        else:
            gender = 0.


        img = load_image(img_path, self.transform)
        
        # read label
        label = self.df_labels.iloc[idx, 0]
        label = np.array([label])

        # label = (label - mu) / sigma
        
        sample = {'images': img, 'labels': label, 'gender': gender}

        return sample


def load_image(img_path, transform=None):
    img = PIL.Image.open(img_path)

    img = transforms.functional.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC)

    # CLAHE has already done it, comment out that line
    img = np.array(img)
    # img = equalize_adapthist(img)
    img = PIL.Image.fromarray(img)

    
    if transform is not None:
        img = transform(img)

        # albumentations
        # img = transform(image=np.array(img))

    img = transforms.functional.to_tensor(img)

    # albumentations
    # img = as_tensor(img['image'])

    return img

def generate_dataset(male):
    # prepare full dataset
    # Pass in male=True (indicates male), there are 6833 male pictures in total
    # full_dataset = BoneAgeDataset('D:\\YangRui\\adjusted_data\\RSNA\\label\\RSNA_fileName.csv', 'D:\\YangRui\\adjusted_data\\RSNA\\whole_hand\\train500\\', male=True)
    train_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/RSNA_fileName.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/train500', male=male)
    val_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/RSNA_fileName_test.csv','/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/test500',male=male,apply_transform=False)

    # train_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/kmeans_results/train/1_0_2.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/train500', male=male)
    # val_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/kmeans_results/test/adjusted/1_delete.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/test500', male=male, apply_transform=False)

    # train_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/kmeans_results/train/4.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/train500', male=male)
    # val_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/label/kmeans_results/test/adjusted/4.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/RSNA/whole_hand/test500', male=male, apply_transform=False)
    
    # # DHA dataset
    # train_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/label/reading1/DHA_traindata5.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/whole_hand/500/fold1/train', male=male)
    # val_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/label/reading1/DHA_testdata5.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/whole_hand/500/fold1/test', male=male, apply_transform=False)
    
    # DHA-5 groups dataset
    # train_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/label/kmeans_results/DHAf5/train/4.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/whole_hand/500/fold1/train', male=male)
    # val_dataset = BoneAgeDataset('/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/label/kmeans_results/DHAf5/test/4.csv', '/home/ncrc-super/data/wgy/bone/adjusted_data/DHA/whole_hand/500/fold1/test', male=male, apply_transform=False)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=WORKERS)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    return train_dataset, train_dataset, val_dataset, train_loader, val_loader
    
    # return full_dataset, train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':
    # test dataset
    full_mixed_dataset, mixed_train_dataset, mixed_val_dataset, mixed_train_loader, mixed_val_loader = generate_dataset(None)
    print('Dataset length: ', len(full_mixed_dataset))
    print('Full ds item: ', full_mixed_dataset[0]['images'].shape, full_mixed_dataset[0]['labels'].shape)
    
    # test load_image:
    # img = load_image('D:/YangRui/adjusted_data/RSNA/whole_hand/train500/2904.png')
    # img = np.array(img)
    # img = PIL.Image.fromarray(img)
    # img = Image.open(img)
    # img.show()
    # print(img)