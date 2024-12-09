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


class Dummy (nn.Module):

  def __init__(self):
    super(Dummy, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(11, 11), stride=2, padding=5)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(8, 8), stride=2, padding=4)
    self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(8, 8), stride=2, padding=4, output_padding=1)
    self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(11, 11), stride=2, padding=5)
    

  def forward(self, x): # x = [10, 2, 251, 321]
    x = F.relu(self.conv1(x)) # [10, 16, 126, 161]
    x = F.relu(self.conv2(x)) # [10, 32, 64, 81]
    x = F.relu(self.deconv1(x)) #[10, 16, 127, 161]
    x = x[:, :, :126, :] # slicing nécessaire pour que les dimensions correspondent, car la convolution transpose n'est pas l'inverse de la convolution x.shape -> #[10, 16, 127, 161]
    x = F.relu(self.deconv2(x)) #[10, 1, 251, 321]
    return x # on ressort un masque
  
model_name='dummy_snd_test'
if not os.path.exists('./models/'+model_name):
    os.makedirs('./models/'+model_name)

chemin_vers_sauvegarde_model ='./models/'+model_name+'/'

# set train_dummy to True to train the model
train_dummy = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Dummy()
n_epochs=200
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())
model.to(device)
loss_train=[]
loss_val=[]
if train_dummy:
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
            torch.save(model, chemin_vers_sauvegarde_model+model_name+'_'+str(epoch)+'.pth')
        loss_train.append(losstrain/counttrain)
        loss_val.append(lossval/countval)
        
    torch.save(model, chemin_vers_sauvegarde_model+'_final'+'.pth')


    # saving the losses in txt files : 
    loss_list_val = [loss_val[i].detach().cpu().numpy() for i in range(len(loss_val))]
    loss_list_train = [loss_train[i].detach().cpu().numpy() for i in range(len(loss_train))]


    with open('./losses/loss_val_'+model_name+'.txt', 'w') as f : 
        for elt in loss_list_val : 
            f.write(str(elt) + '\n')

    with open('./losses/loss_train_'+model_name+'.txt', 'w') as f :
        for elt in loss_list_train : 
            f.write(str(elt) + '\n')