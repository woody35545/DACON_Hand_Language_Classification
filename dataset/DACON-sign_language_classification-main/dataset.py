from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None): #필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def stacked_channel(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = 0
        img2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        normalized_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_hsv = cv2.normalize(hsv, None, 0, 255, cv2.NORM_MINMAX)
        normalized_img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        stacked_img = np.concatenate((normalized_img, normalized_hsv, normalized_img2), axis=2)
        # print(stacked_img.shape)
        return stacked_img

    def augmentation(self, image):
        p = np.random.rand(1)
        if p > 0.5:
            image = cv2.flip(image, 1)

        def rotation(img, angle):
            angle = int(np.random.uniform(-angle, angle))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            return img
        image = rotation(image, 90)
        return image

    def __getitem__(self, index): #index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.train_mode:
            image = self.augmentation(image)
        # return image
        # image = self.transforms(image)
        # print(image)
        # print(image.shape)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = 0
        img2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        laplacian = laplacian[:, :, np.newaxis]
        laplacian = np.concatenate([laplacian, laplacian, laplacian], axis=2)
        
        normalized_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_hsv = cv2.normalize(hsv, None, 0, 255, cv2.NORM_MINMAX)
        normalized_img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        # stacked_img = np.concatenate((normalized_img, normalized_hsv, normalized_img2), axis=2)
        # stacked_img = torch.tensor(stacked_img)
        # image = self.stacked_channel(image)
        # if self.transforms is not None:
        image = self.transforms(normalized_img)
        hsv = self.transforms(normalized_hsv)
        image2 = self.transforms(normalized_img2)
        laplacian = self.transforms(laplacian)
        laplacian = laplacian[0, :, :].unsqueeze(0)
        # print(laplacian.shape)
        # exit(0)
        stacked_image = torch.cat((image, hsv, image2, laplacian), dim=0)
        # stacked_image = torch.cat((image, hsv, image2), dim=0)
        # print(stacked_image.shape)

        if self.train_mode:
            label = self.label_list[index]
            return stacked_image, label
        else:
            return stacked_image
    
    def __len__(self): #길이 return
        return len(self.img_path_list)