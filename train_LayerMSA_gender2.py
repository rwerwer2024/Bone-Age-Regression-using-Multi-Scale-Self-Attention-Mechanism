
EPOCHS = 50
GRAD_NORM_CLIP = 0.1
LEARNING_RATE = 1e-5
GENDER_SENSITIVE = True
# GENDER_SENSITIVE = False

# trained model path to be loaded
LOAD_MIXED = None
LOAD_MALE = None
LOAD_FEMALE = None
LAMBDA = 0.5


# In[10]:
# import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='./')

from operator import mod
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import collections
import datetime
import math
import numpy as np # linear algebra
import tqdm
import pandas as pd
import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet50




# from tensorboardX import SummaryWriter

#from models.vgg import vgg16_bn as Model
# from models.resnet import resnet50 as Model
#from models.mnasnet import mnasnet1_0 as Model

# from model import Res_Vit_B16 as Model,BasicBlock
# from model2 import Res_Vit_B16 as Model,BasicBlock
# from model_LayerMSA_gender import Res_Vit_B16_gender_512 as Model,BasicBlock
# from model_LayerMSA_gender import Res_Vit_B16_gender_512_new as Model,BasicBlock
# from model_LayerMSA_gender_test import  Res34_LCFF_V1_old, Res50_LCFF_V1_SA, Res_Vit_B16_gender_512_new as Model,BasicBlock,Bottleneck
# from model_LayerMSA_gender_test import Res_Vit_B16_gender_512_sa1 as Model,BasicBlock,Bottleneck
# from model_LayerMSA_gender_test import Res_Vit_B16_gender_512_sa1_bottleneck as Model,BasicBlock,Bottleneck
from model_LayerMSA_gender_test import Res101_LCFF_V1, Res50_LCFF_V1_n64, Res34_LCFF_V1, Res18_LCFF_V1, Res34_LCFF_V1_n256, Res34_LCFF_V1_n16, Res34_LCFF_V1_old, Res34_LCFF_V1_old_stn
from model_LayerMSA_gender_HignerOrder import Res34_LCFF_V1_old_highorder 

from radam_optimizer import RAdam
from triplet_loss import triplet_loss, TripletLoss, TripletLoss2, TripletLoss3, AdapitiveTripletLoss

# from dataset_gender import load_image, generate_dataset
from dataset_gender_tripletloss import load_image, generate_dataset

# module to notify training status
# from fcm_notifier import FCMNotifier
# notifier = FCMNotifier()



if not GENDER_SENSITIVE:
    # prepare full dataset
    full_mixed_dataset, mixed_train_dataset, mixed_val_dataset, mixed_train_loader, mixed_val_loader = generate_dataset(None)
    print('Dataset length: ', len(full_mixed_dataset))
    print('Full ds item: ', full_mixed_dataset[0]['images'].shape, full_mixed_dataset[0]['labels'].shape)

else:
    # prepare male dataset
    full_male_dataset, male_train_dataset, male_val_dataset, male_train_loader, male_val_loader = generate_dataset(True)
    print('Male dataset length: ', len(full_male_dataset))
    print('Male ds item: ', full_male_dataset[0]['images'].shape, full_male_dataset[0]['labels'].shape)

    # prepare female dataset
    full_female_dataset, female_train_dataset, female_val_dataset, female_train_loader, female_val_loader = generate_dataset(False)
    print('Female dataset length: ', len(full_female_dataset))
    print('Female ds item: ', full_female_dataset[0]['images'].shape, full_female_dataset[0]['labels'].shape)


# # Train

# In[12]:
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/resnet152.pth'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/resnet101.pth'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/resnet50.pth'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/resnet34.pth'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/resnet18.pth'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/1000-256-1_group3_3.812.pt'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/1000-256-1_group0_1.623.pt'
# weight_path = '/home/ncrc-super/data/wgy/pretrained_models/res34_group0_bs2.pt'

weight_path = '/home/ncrc-super/data/wgy/Res_Vit/high_order/model_3463_4.765.pt'

def generate_model():
    print('Generating model')
    # model = Model(block=BasicBlock,layers=[2,2,2,2],in_channels=3,)

    # res101
    # model = Model(block=Bottleneck,layers=[3,4,23,3],in_channels=3,)

    # model = Model(block=Bottleneck,layers=[3,4,6,3],in_channels=3,depth=0)
    # model = Res34_LCFF_V1()
    model = Res34_LCFF_V1_old()
    # model = Res34_LCFF_V1_old_highorder()
    # model = Res34_LCFF_V1_old_stn()
    # model = resnet50()


    # model = Model(block=Bottleneck,layers=[3,4,6,3],in_channels=3,)

    # res152
    # model = Model(block=Bottleneck,layers=[3,8,36,3],in_channels=3,)
    # model.cuda()
    # print(model)

    # 使用自己预训练模型
    # pre_dict = torch.load(weight_path)
    # model = pre_dict['model']
    # model = pre_dict

    # 使用resnet在imagenet上的权重的时候删除最后的fc参数
    # pre_dict = torch.load(weight_path)
    # del_keys = ['fc.weight','fc.bias']
    # for key in del_keys:
    #     del pre_dict[key]
    # model.load_state_dict(pre_dict,strict=False)

    # 使用自己训练的checkpoint文件
    # checkpoint = torch.load(weight_path)
    # model = checkpoint['model']
    # model.cuda()

    # tl = TripletLoss2(m = 0.5)
    # tl = TripletLoss3()
    tl = AdapitiveTripletLoss()
    # tl_self = TripletLoss_self(margin=0.5)

    # optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': tl.parameters()},
    ], lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    model.cuda()
    
    return model, optimizer, scheduler

def load_model(path):
    print('Loading model from ', path)
    checkpoint = torch.load(path)
    print('Loaded ' + path + ' on epoch', checkpoint['epoch'], 'train loss:', checkpoint['train_loss'], 'and val loss: ', checkpoint['val_loss'])
    return checkpoint['model'], checkpoint['optimizer'], checkpoint['scheduler']




if not GENDER_SENSITIVE:
    # full mixed (male/female) model
    if LOAD_MIXED is None:
        mixed_model, mixed_optimizer, mixed_scheduler = generate_model()
    else:
        mixed_model, mixed_optimizer, mixed_scheduler = load_model(LOAD_MIXED)
    # print(mixed_model)
    
else:
    # male model
    if LOAD_MALE is None:
        male_model, male_optimizer, male_scheduler = generate_model()
    else:
        male_model, male_optimizer, male_scheduler = load_model(LOAD_MALE)
        
    # female model
    if LOAD_FEMALE is None:
        female_model, female_optimizer, female_scheduler = generate_model()
    else:
        female_model, female_optimizer, female_scheduler = load_model(LOAD_FEMALE)
    
    # print(male_model) # print only one since they're equal




def save_model(experiment_name, model, optimizer, scheduler, epoch, train_loss, val_loss):
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    # torch.save(checkpoint, 'D:\\work\\Bone_Resnet\\trained_models' + experiment_name + '_' + str(datetime.datetime.now()) + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/trained_models/' + experiment_name + '_LayerMSA_newmodel_512_gender_layernorm_1000-1024-1' + '.pt')
    
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/trained_models/' + experiment_name + 'group4_bs2_4' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/new_models/res152/' + experiment_name + 'group1_bs2_4__1_0_2trans__1_deletetest' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/new_models/DHA/' + experiment_name + 'fold1_bs2_4' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/ablation/RSNA/' + experiment_name + '_RSNA_withoutPSA_Res34(3)' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/modified/no_patch_embedding/' + experiment_name + '_RSNA_withtPSA_Res50_8head_noposition' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/modified/patch_embedding/' + experiment_name + '_RSNA_withtPSA_Res18_8head_noposition' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/ablation/RSNA/LCFF_V1_withPSA_n16/' + experiment_name + '_RSNA_withtPSA_Res34_8head_noposition_n16' + '.pt')
    # torch.save(checkpoint, '/home/ncrc-super/data/wgy/Res_Vit/ablation/RSNA/LCFF_V1_retrain/' + experiment_name + '_RSNA_4.733_retrain' + '.pt')
    print('Model ' + experiment_name + ' saved.')

y1 = []
y2 = []
x1 = [i+1 for i in range(EPOCHS)]    
    
def train(experiment_name, model, optimizer, scheduler, train_loader, val_loader, epochs=EPOCHS):
    # train_loss_hist = collections.deque(maxlen=500)
    # val_loss_hist = collections.deque(maxlen=500)

    # train_loss_hist = []
    # val_loss_hist = []

    # model.train()
    
    best_model = None
    best_val_loss = 1e6
    best_val_mae = 100
    
    loss_fn = nn.MSELoss()
    metrics = nn.L1Loss()
    
    # writer = SummaryWriter()
    # tl = TripletLoss2(m = 0.5)
    # tl = TripletLoss3()
    tl = AdapitiveTripletLoss()

    for epoch_num in range(epochs):
        model.train()

        train_loss_hist = []
        val_loss_hist = []
        epoch_loss = []

        L1_hist = []
        val_L1_hist = []


        # 参数：total：总的项目数
        #       desc：进度条左边的描述文字
        #       ncols：进度条的宽度
        progress = tqdm.tqdm(total=len(train_loader), desc='Training Status', position=0)
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()

            # preds = model(data['images'].cuda().float())
            # preds = model(data['images'].cuda().float(),data['gender'].cuda().float())
            preds, minibatch_features = model(data['images'].cuda().float(),data['gender'].cuda().float())

            loss_triplet = tl(minibatch_features, data['labels'].cuda().float())

            loss = loss_fn(preds, data['labels'].cuda().float())

            if loss_triplet == 0:
                loss += loss_triplet
            else:
                # loss = loss * 0.9
                loss += loss_triplet.squeeze(0) * LAMBDA


            L1 = metrics(preds,data['labels'].cuda().float())
            # loss = loss.mean()

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)

            optimizer.step()

            loss = float(loss)
            
            train_loss_hist.append(loss)
            L1_hist.append(L1.item())

            epoch_loss.append(loss)
            
            progress.set_description(
                desc='Train - Ep: {} | It: {} | Ls: {:1.3f} | mLs: {:1.3f} | MAE: {:1.3f}'.format(
                    epoch_num, 
                    iter_num, 
                    loss, 
                    np.mean(train_loss_hist),
                    L1.item()
                    # math.sqrt(loss)
                )
            )
            
            progress.update(1)
            
            del loss

        train_loss = np.mean(train_loss_hist)
        # train_mae = math.sqrt(train_loss)
        train_mae = np.mean(L1_hist)
        y1.append(train_mae)
        
        # writer.add_scalar('loss/train_loss_mean', train_loss, epoch_num)
        # writer.add_scalar('loss/train_mae', train_mae, epoch_num)
        
        print('Train - Ep: {} | Ls: {:1.3f} | MAE: {:1.3f}'.format(epoch_num, train_loss, train_mae))
        
        progress = tqdm.tqdm(total=len(val_loader), desc='Validation Status', position=0)

        # model.eval()
        val_loss_total = 0
        for iter_num, data in enumerate(val_loader):
            with torch.no_grad():
                
                
                # preds = model(data['images'].cuda().float())
                # preds = model(data['images'].cuda().float(),data['gender'].cuda().float())
                preds, _ = model(data['images'].cuda().float(),data['gender'].cuda().float())       # triplet_loss
                val_loss = loss_fn(preds, data['labels'].cuda().float())
                val_L1 = metrics(preds,data['labels'].cuda().float())
                # val_loss.mean()
                val_loss_hist.append(float(val_loss))
                val_L1_hist.append(val_L1.item())
                # val_loss_total += float(val_loss)
                optimizer.zero_grad()

                # 为显示的条条指定描述信息
                progress.set_description(
                    desc='Val - Ep: {} | It: {} | Ls: {:1.5f} | mLs: {:1.5f} | MAE: {:1.3f}'.format(
                        epoch_num, 
                        iter_num, 
                        float(val_loss), 
                        np.mean(val_loss_hist),
                        val_L1.item()
                        # math.sqrt(float(val_loss))
                    )
                )
                # 下面的条条每次更新多少
                progress.update(1)

                del val_loss

        # val_loss = np.mean(val_loss_hist)
        # val_mae = math.sqrt(val_loss)

        # val_loss_eopch = val_loss_total / 10
        val_loss = np.mean(val_loss_hist)
        val_mae = np.mean(val_L1_hist)
        y2.append(val_mae)

        # writer.add_scalar('loss/val_loss', val_loss, epoch_num)
        # writer.add_scalar('loss/val_mae', val_mae, epoch_num)

        # val_mae = math.sqrt(val_loss)
        
        # writer.add_scalar('loss/val_loss', val_loss, epoch_num)
        # writer.add_scalar('loss/val_mae', val_mae, epoch_num)
        
        # notifier.notify(Epoch=epoch_num, Train_MAE=train_mae, Val_MAE=val_mae)
        print('Val - Ep: {} | Ls: {:1.5f} | MAE: {:1.3f}'.format(epoch_num, val_loss, val_mae))

        scheduler.step(np.mean(epoch_loss))
        
        # if val_loss < best_val_loss:
        #     save_model('checkpoint_' + experiment_name, model, optimizer, scheduler, epoch_num, train_loss, val_loss)
        #     # best_val_loss = val_loss
        #     best_val_loss = val_loss
        #     best_model = copy.deepcopy(model)
        
        # if val_mae < best_val_mae:
        #     # torch.save(model, "./LCFF34_tripletloss.pt")
        #     torch.save(model, "./High_order=3_lambda0.2.pt")
        #     print("--------------------model-saved--------------------")
        #     # save_model('checkpoint_' + experiment_name, model, optimizer, scheduler, epoch_num, train_loss, val_loss)
        #     # best_val_loss = val_loss
        #     best_val_mae = val_mae
        #     best_model = copy.deepcopy(model)
        
        if val_mae < 5.10 and val_mae > 5.05:
            torch.save(model, "./High_order=3_lambda0.5.pt")
            print("--------------------model-saved--------------------")
            best_model = copy.deepcopy(model)
            break

    model.eval()
    # save_model('_final_' + experiment_name, model, optimizer, scheduler, epochs - 1, np.mean(train_loss_hist), np.mean(val_loss_hist))
    
    # writer.close()
    
    return best_model, model, optimizer, scheduler


# In[8]:


if not GENDER_SENSITIVE:
    print('\nTRAINING MIXED MODEL')
    mixed_model, _, mixed_optimizer, mixed_scheduler = train('mixed', mixed_model, mixed_optimizer, mixed_scheduler, mixed_train_loader, mixed_val_loader, EPOCHS)
else:
    print('\nTRAINING MALE MODEL')
    male_model, _, male_optimizer, male_scheduler = train('male', male_model, male_optimizer, male_scheduler, male_train_loader, male_val_loader, EPOCHS)
    print('\nTRAINING FEMALE MODEL')
    female_model, _, female_optimizer, female_scheduler = train('female', female_model, female_optimizer, female_scheduler, female_train_loader, female_val_loader, EPOCHS)

# fig = plt.figure(figsize=(8,6))
# plt.plot(x1,y1,y2,'b-')
# plt.xlabel('epoch')
# plt.ylabel('L1_loss')
# plt.savefig('/home/ncrc-super/data/wgy/Res_Vit/adjusted/group2_256_2222_34sa.png')
# plt.show()

# writer.close()
# print('tensorboard --logdir=./ --port 12349')