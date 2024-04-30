import h5py
import torch
import torch.utils.data as data
import torch.nn as nn

from torch.utils.data import SubsetRandomSampler
# from peer_net_v2 import PEERnet,trainer,valer
from my_resnet_peer import resnet5, trainer, valer, resnet10,resnet18
from eye_data import MyData
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ########## set hyperparameters

    train_batch_size = 16
    LR = 0.00001
    epochs = 50
    data_path = "/data2/xiuwen/new_data_final/"
    model_path = "/data2/xiuwen/eyemovent/eye_models_final/raw_eye_x_partici_correct_65.pt"


    ########## load data

    train_dataset = MyData(data_path,'train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)

    val_dataset = MyData(data_path,'test')
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=75, shuffle=False, pin_memory=True)


    ########## load model and setting model

    net = resnet10()
    net_p = nn.DataParallel(net.cuda(device=3), device_ids=[3,4, 5, 6,7])
    # net.load_state_dict(torch.load(model_path))
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net_p.parameters(), lr=LR, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.8)
    #
    #     val_loss_curv = []
    for epoch in range(epochs):

        train_loss = trainer(net_p, train_loader, optimizer, epoch, loss_func)
        val_loss,out,labels = valer(net_p,val_loader,loss_func)


        # scheduler.step()
        #torch.save(net_p.state_dict(), '/data2/xiuwen/eyemovent/res5_eye_models_final/raw_eye_y_partici_correct_{}.pt'.format(str(epoch)))
        # if epoch >25:
        #     torch.save(net.state_dict(), '/data2/xiuwen/eyemovent/res5_eye_models_final/res5_raw_eye_x_partici_correct_{}.pt'.format(str(epoch)))

    # plt.plot(out)
    # plt.plot(labels)
    # plt.show()

