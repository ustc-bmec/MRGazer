import h5py
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import random_split
#from peer_net_v2 import PEERnet,trainer,valer
from resnet_peer import resnet5, trainer,valer,resnet10
import matplotlib.pyplot as plt
import os



class Load_data(data.Dataset):

    def __init__(self,path):
        super(Load_data, self).__init__()
        h5f = h5py.File(path,'r')
        self.img = h5f['eye']
        self.label = h5f['labels']
        self.centroid = h5f['centroid']
        self.bbox = h5f['bbox']

    def __getitem__(self,idx):
        image = torch.from_numpy(self.img[idx]).float()
        label = torch.from_numpy(self.label[idx]).float()
        centroid = torch.from_numpy(self.centroid[idx]).float()
        bbox = torch.from_numpy(self.bbox[idx]).float()


        return image,label,centroid,bbox

    def __len__(self):
        return self.img.shape[0]


if __name__ == '__main__':

    train_batch_size = 128
    LRS = [0.0005]
    # LR = 0.00001
    epochs = 70
    path = 'train_data.h5'
    dataset = Load_data(path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)

    val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True)
    net = resnet10()
    #net = PEERnet()
    # net = nn.DataParallel(net,device_ids=[0,1,2])
    net_p = nn.DataParallel(net.cuda(),device_ids=[0,1])
    loss_func = torch.nn.MSELoss()
    target_val_loss = 5.7
    for LR in LRS:
        print(LR)
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)
    
        train_loss_curv = []
        val_loss_curv = []
        for epoch in range(epochs):
            train_loss = trainer(net_p, train_loader, optimizer, epoch, loss_func)
            scheduler.step()
            train_loss_curv.append(train_loss)
            val_loss = valer(net_p, val_loader, loss_func)
            val_loss_curv.append(val_loss)
            if val_loss <= target_val_loss:
                print("save model")
                torch.save(net_p.state_dict(), 'raw_eye_x.pt')
                break
        plt.plot(train_loss_curv, label='train_loss')
        plt.plot(val_loss_curv, label='val_loss', color='red')
        plt.savefig('trial_loss_curve_' + str(LR) + '.jpg')
        plt.clf()



