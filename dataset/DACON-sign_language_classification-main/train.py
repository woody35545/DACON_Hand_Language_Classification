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

import torchvision.datasets import ImageFolder

CFG = {
    'IMG_SIZE':224, #이미지 사이즈
    'EPOCHS':400, #에포크
    'LEARNING_RATE':0.001, #학습률
    'BATCH_SIZE':12, #배치사이즈
    'SEED':41, #시드
    'DECAY_MARGIN':0.2, # Decay margin
    'LR_RATE':0.3, # Decay rate
    'DECAY_START':False, # Decay rate
}

model = models.efficientnet_b3(pretrained=False)
out_channel = model.features[0][0].out_channels
model.features[0][0] = nn.Conv2d(10, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# print(model.features[0][0])
# exit(0)
model.fc = nn.Linear(1000, 11)
torch.nn.init.xavier_uniform_(model.fc.weight)

model = model.to(device)
# checkpoint = torch.load('./saved/best_model.pth')
# model.load_state_dict(checkpoint)
criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

optimizer = torch.optim.Adam(params=model.parameters(), lr= CFG["LEARNING_RATE"] )#0.001
scheduler = None


from dataset import CustomDataset

# Seed 고정
import random
import numpy as np
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])


from glob import glob

def get_train_data(data_dir):
    img_path_list = []
    label_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    #print('wow', img_path_list)
        
    # get label
    #label_df = pd.read_csv(data_dir+'/train.csv')
    label_list.extend(label_df['label'])
    #print('wow2', label_list)
                
    return img_path_list, label_list

def get_test_data(data_dir):
    img_path_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    # print(img_path_list)
    
    return img_path_list

def train(model, optimizer, train_loader, vali_loader, scheduler, device): 
    model.to(device)
    n = len(train_loader)
    
    #Loss Function 정의
    #criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0
    
    for epoch in range(1,CFG["EPOCHS"]+1): #에포크 설정
        model.train() #모델 학습
        running_loss = 0.0
            
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device) #배치 데이터
            optimizer.zero_grad() #배치마다 optimizer 초기화
        
            # Data -> Model -> Output
            logit = model(img) #예측값 산출
            loss = criterion(logit, label) #손실함수 계산
            
            # 역전파
            loss.backward() #손실함수 기준 역전파 
            optimizer.step() #가중치 최적화
            running_loss += loss.item()
                
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        
        if scheduler is not None:
            scheduler.step()
            
        #Validation set 평가
        model.eval() #evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        with torch.no_grad(): #파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  #11개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item() #예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vali_loader), correct, len(vali_loader.dataset), 100 * correct / len(vali_loader.dataset)))
        
        if vali_loss < CFG["DECAY_MARGIN"] and CFG["DECAY_START"] == False:     # If validation loss becomes lower than the decay margin
            CFG["DECAY_START"] = True
            optimizer = torch.optim.Adam(params=model.parameters(), lr= CFG["LEARNING_RATE"] * CFG["LR_RATE"] )#0.0001
        #베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(), './saved/best_model.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')

    print("best accuracy:", best_acc)

if __name__ == '__main__':

    label_df = pd.read_csv('data/train.csv')
    label_df.head()

    label_df['label'][label_df['label'] == '10-1'] = 10 ## label : 10-1 -> 10
    label_df['label'][label_df['label'] == '10-2'] = 0 ## Label : 10-2 -> 0
    label_df['label'] = label_df['label'].apply(lambda x : int(x)) ## Dtype : object -> int

    all_img_path, all_label = get_train_data('data/train')
    test_img_path = get_test_data('data/test')

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
                transforms.ToPILImage(), #이미지 데이터를 tensor
                transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]), #이미지 사이즈 변형
                transforms.ToTensor(), #이미지 데이터를 tensor
                transforms.Normalize(img_mean, img_std)
                ])

    # Train : Validation = 0.8 : 0.25 Split
    train_len = int(len(all_img_path)*0.75)
    Vali_len = int(len(all_img_path)*0.25)

    train_img_path = all_img_path[:train_len]
    train_label = all_label[:train_len]

    vali_img_path = all_img_path[train_len:]
    vali_label = all_label[train_len:]

    print('train set 길이 : ', train_len)
    print('vaildation set 길이 : ', Vali_len)

    # Get Dataloader
    #CustomDataset class를 통하여 train dataset생성
    train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=transform) 
    #만든 train dataset를 DataLoader에 넣어 batch 만들기
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0) #BATCH_SIZE : 12

    #vaildation 에서도 적용
    vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=transform)
    vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    train_batches = len(train_loader)
    vali_batches = len(vali_loader)

    print('total train imgs :',train_len,'/ total train batches :', train_batches)
    print('total valid imgs :',Vali_len, '/ total valid batches :', vali_batches)

    train(model, optimizer, train_loader, vali_loader, scheduler, device)
