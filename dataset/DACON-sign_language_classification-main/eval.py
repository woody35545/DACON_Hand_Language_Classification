import os 
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당
#하이퍼 파라미터 튜닝
import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset

from torchvision import models
# from torchvision.models import efficientnet_b3 as efficientnet
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm
import torchsummary
import pandas as pd

from glob import glob
from dataset import CustomDataset

CFG = {
    'IMG_SIZE':224, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':0.001, #학습률
    'BATCH_SIZE':8, #배치사이즈
    'SEED':41, #시드
}

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

checkpoint = torch.load('./saved/best_model.pth')
model = models.efficientnet_b3(pretrained=False)
model.features[0][0] = nn.Conv2d(10, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.fc = nn.Linear(1000, 11)
model = model.to(device)
model.load_state_dict(checkpoint)

def get_test_data(data_dir):
    img_path_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    # print(img_path_list)
    
    return img_path_list
    
def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred

test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
                    transforms.ToTensor(),
                    transforms.Normalize(img_mean, img_std)
                    ])

test_img_path = get_test_data('data/test')
test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# Inference

if __name__ == '__main__':
    preds = predict(model, test_loader, device)

    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    submission['label'] = submission['label'].apply(lambda x : str(x)) ## Dtype : int -> object
    # print(submission['label'][submission['label'] == 10])
    print(submission)
    submission['label'][submission['label'] == '10'] = '10-1' ## label : 10 -> '10-1'
    submission['label'][submission['label'] == '0'] = '10-2' ## Label : 0 -> '10-2'
    # submission['label'] = submission['label'].apply(lambda x : str(x)) ## Dtype : int -> object
    submission.head()
    submission.to_csv('submit2.csv', index=False)

