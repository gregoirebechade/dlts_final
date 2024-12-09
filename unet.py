import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
from scipy.io import wavfile
from torch.utils.data import DataLoader
import tqdm
from scipy.signal import stft, istft









class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.X = [elt for elt in os.listdir(path_to_data + 'noisy/') if 'npy' in elt]
        self.Y = [elt for elt in os.listdir(path_to_data + 'origin/') if 'npy' in elt]


    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        noised=np.load(self.path_to_data + 'noisy/' + self.X[idx])
        noised_real=noised.real
        noised_imag=noised.imag
        origin=np.load(self.path_to_data + 'origin/' + self.Y[idx])
        origin_real=origin.real
        origin_imag=origin.imag
        return torch.tensor(np.array([noised_real, noised_imag])), torch.tensor(np.array([origin_real, origin_imag])) # 1, 129, 357 chacuns 

dataloader_test = DataLoader(Mydataset('./data/spectrogrammes/test/'), batch_size=10 , shuffle=True)
dataloader_train = DataLoader(Mydataset('./data/spectrogrammes/train/'), batch_size=10 , shuffle=True)
dataloader_validation = DataLoader(Mydataset('./data/spectrogrammes/validation/'), batch_size=10 , shuffle=True)

# class UNet(nn.Module): 
#   def __init__(self):
#     super(UNet, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=2)
#     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2)
#     self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2)
#     self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2)
#     self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=2)
#     self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(6, 6), stride=2)
#     self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(6, 6), stride=2)
#     self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(6, 6), stride=2)
#     self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(6, 6), stride=2)
#     self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=2)

#   def forward(self, x): 
#     x = (self.conv1(x))
#     x = nn.BatchNorm2d(32)(x)
#     x = nn.LeakyReLU(negative_slope=0.2)(x)
#     x = (self.conv2(x))
#     x = nn.BatchNorm2d(64)(x)
#     x = nn.LeakyReLU(negative_slope=0.2)(x)
#     x = (self.conv3(x))
#     x = nn.BatchNorm2d(128)(x)
#     x = nn.LeakyReLU(negative_slope=0.2)(x)
#     x = (self.conv4(x))
#     x = nn.Dropout(p=0.5)(x)
#     x = nn.BatchNorm2d(256)(x)
#     x = (self.conv5(x))
#     x = nn.Dropout(p=0.5)(x)
#     x = nn.BatchNorm2d(512)(x)
#     x = (self.deconv1(x))
#     x = x[: , : , : , :17 ]
#     x = nn.BatchNorm2d(256)(x)
#     x = F.relu(x)
#     x = nn.Dropout(p=0.5)(x)
#     x = (self.deconv2(x))
#     x = x[: , : , : , :37 ]
#     x = nn.BatchNorm2d(128)(x)
#     x = F.relu(x)
#     x = nn.Dropout(p=0.5)(x)
#     x = (self.deconv3(x))
#     x = nn.BatchNorm2d(64)(x)
#     x = F.relu(x)
#     x = nn.Dropout(p=0.5)(x)
#     x = (self.deconv4(x))
#     x = nn.BatchNorm2d(32)(x)
#     x = F.relu(x)
#     x = (self.deconv5(x))
#     x = x[:, :, :, :321]
#     x = F.sigmoid(x)
#     return x 


class UNet(nn.Module): 
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(6, 6), stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(6, 6), stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(6, 6), stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(6, 6), stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn_deconv1 = nn.BatchNorm2d(256)
        self.bn_deconv2 = nn.BatchNorm2d(128)
        self.bn_deconv3 = nn.BatchNorm2d(64)
        self.bn_deconv4 = nn.BatchNorm2d(32)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv4(x)
        x = F.dropout(x, p=0.5)
        x = self.bn4(x)

        x = self.conv5(x)
        x = F.dropout(x, p=0.5)
        x = self.bn5(x)

        x = self.deconv1(x)
        x = x[:, :, :, :17]
        x = self.bn_deconv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.deconv2(x)
        x = x[:, :, :, :37]
        x = self.bn_deconv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.deconv3(x)
        x = self.bn_deconv3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.deconv4(x)
        x = self.bn_deconv4(x)
        x = F.relu(x)

        x = self.deconv5(x)
        x = x[:, :, :, :321]
        x = torch.sigmoid(x)
        return x




chemin_vers_sauvegarde_unet ='./models/unet/unet_final'


# set train_unet to True to train the model
train_unet = True
model_name='unet'
if not os.path.exists('./models/'+model_name):
    os.makedirs('./models/'+model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = UNet()
n_epochs=1
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())
model.to(device)
loss_train=[]
loss_val=[]
if train_unet:
    for epoch in (range(n_epochs)):
        print(epoch)
        losstrain=0
        counttrain=0
        lossval=0
        countval=0
        for batch_x,batch_y in dataloader_train:
            counttrain+=1
            batch_x=batch_x.to(device)
            batch_y = batch_y.long()
            batch_y=batch_y.to(device)
            optimizer.zero_grad()
            mask_predicted = model(batch_x.float())
            batch_y_predicted = batch_x * mask_predicted
            l = loss(batch_y_predicted, batch_y)
            l.backward()
            losstrain+=l
            optimizer.step()
        for batch_x,batch_y in dataloader_validation:
            countval+=1
            batch_x=batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.no_grad():
                mask_predicted = model(batch_x.float())
                batch_y_predicted =  batch_x * mask_predicted
                l = loss(batch_y_predicted, batch_y)
                lossval+=l
        if epoch%10==0:
            print(f'epoch {epoch}, training loss = {losstrain/counttrain}')
            torch.save(model, chemin_vers_sauvegarde_unet+'_'+str(epoch)+'.pth')
        loss_train.append(losstrain/counttrain)
        loss_val.append(lossval/countval)
        
    torch.save(model, chemin_vers_sauvegarde_unet+'.pth')


    # saving the losses in txt files : 
    loss_list_val = [loss_val[i].detach().cpu().numpy() for i in range(len(loss_val))]
    loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]


    with open('./losses/loss_val_'+model_name+'.txt', 'w') as f : 
        for elt in loss_list_val : 
            f.write(str(elt) + '\n')

    with open('./losses/loss_train_'+model_name+'.txt', 'w') as f :
        for elt in loss_list_train : 
            f.write(str(elt) + '\n')