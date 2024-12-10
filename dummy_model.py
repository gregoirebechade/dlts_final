import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
from scipy.io import wavfile
from torch.utils.data import DataLoader
import tqdm
from scipy.signal import stft, istft




class Dummy_model(nn.Module): 
    def __init__(self): 
        super(Dummy_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1,  kernel_size=11, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,  kernel_size=11, stride=1, padding='same')
    def forward(self, x): # [10, 2, 251, 321]
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2) # [10, 1, 251, 321]
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x
    


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



chemin_vers_sauvegarde_dummy = 'models/dummy_model/'


# set train_dummy to True to train the model
train_dummy_model = True

model_name='dummy_model'
if not os.path.exists('models/'+model_name):
    os.makedirs('models/'+model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Dummy_model()
n_epochs=200
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())
model.to(device)
loss_train=[]
loss_val=[]
if train_dummy_model:
    for epoch in (range(n_epochs)):
        print(epoch)
        losstrain=0
        counttrain=0
        lossval=0
        countval=0
        for batch_x,batch_y in dataloader_train:
            print(model[batch_x.float()])
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
            print(model.parameters())
            torch.save(model, chemin_vers_sauvegarde_dummy+model_name+'_'+str(epoch)+'.pth')
        loss_train.append(losstrain/counttrain)
        loss_val.append(lossval/countval)
        
    torch.save(model, chemin_vers_sauvegarde_dummy+'_final'+'.pth')


    # saving the losses in txt files : 
    loss_list_val = [loss_val[i].detach().cpu().numpy() for i in range(len(loss_val))]
    loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]


    with open('./losses/loss_val_'+model_name+'.txt', 'w') as f : 
        for elt in loss_list_val : 
            f.write(str(elt) + '\n')

    with open('./losses/loss_train_'+model_name+'.txt', 'w') as f :
        for elt in loss_list_train : 
            f.write(str(elt) + '\n')



