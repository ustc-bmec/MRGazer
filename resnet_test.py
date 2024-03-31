from resnet_peer import resnet10
import torch
from main_peer import Load_data
import torch.utils.data as data
from resnet_peer import convert_to_pix
import matplotlib.pyplot as plt
import torch.nn as nn



def cal_sum_loss(loss):

    sqrt_loss = torch.sqrt(loss)
    batch_sum_loss = torch.sum(sqrt_loss)
    one_batch_loss = batch_sum_loss/108
    return one_batch_loss

def test(net,test_loader,func):
    net.eval()
    with torch.no_grad():
        sum_loss = 0.0
        for id, (img, labels, _, _) in enumerate(test_loader):
            img = img.unsqueeze(1)
            img = img.cuda()
            labels = convert_to_pix(labels)
            labels = labels.cuda()
            results = net(img)
            loss = func(results,labels)
            one_batch_loss = cal_sum_loss(loss)
            sum_loss += one_batch_loss.item()
            plot_pic(results,labels,id)
        sum_loss = sum_loss/(id+1)

    return sum_loss

def plot_pic(results,labels,id):

    times = [i+1 for i in range(108)]
    results = results.cpu()
    labels = labels.cpu()
    results = results.numpy()
    labels = labels.numpy()
    plt.plot(times,results,color = 'red',label = 'results')
    plt.plot(times,labels,color='blue',label = 'labels')
    plt.xlabel('times')
    plt.ylabel('fixation/°')
    plt.legend()
    plt.savefig('picture/results_' + str(id+1) + '.jpg')
    plt.clf()


if __name__ == '__main__':

    data_path = 'test_net_final.h5'
    model_path = 'raw_eye_x.pt'
    test_dataset = Load_data(data_path)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=108, shuffle=False, pin_memory=True)
    net = resnet10()
    net_p = nn.DataParallel(net.cuda(),device_ids=[0,1])
    net_p.load_state_dict(torch.load(model_path))
    loss_func = torch.nn.MSELoss(reduction='none')
    sum_loss = test(net_p,test_loader,loss_func)
    print('the final mae loss =%.3f'%sum_loss)

