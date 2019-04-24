from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision.datasets import mnist as mn
from torchvision import datasets, transforms



def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))  # 维度转化
    x = torch.from_numpy(x)
    return x

# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize([0.5], [0.5])])

train_set = mn.MNIST('data', train=True, transform=data_tf, download=True)
test_set = mn.MNIST('data', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)

test_data = DataLoader(test_set, batch_size=128, shuffle=False)
print(len(train_data), '******', len(test_data))

a, a_label = next(iter(train_data))
# a, a_label = next(iter(train_data))

# net = nn.Sequential(
#     nn.Linear(784, 300),  # 因为28*28=748
#     nn.ReLU(),
#     nn.Linear(300, 10)  # 最后输出10个分类
# )
net = nn.Sequential(
    nn.Linear(784, 400), #  因为28*28=748
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10) #  最后输出10个分类
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-2)  # 学习率0.1


losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data[0]
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct.item() / im.shape[0]  # num_correct.item()
        train_acc += acc
        # _, pred = torch.max(out, 1)  # _, pred = out.max(1)
        # num_correct = (pred == label).sum()  # num_correct = (pred == label).sum().data[0]
        # train_acc += num_correct.item()

    # losses.append(train_loss / len(train_set))
    # acces.append(train_acc / len(train_set)

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label) # 交叉熵计算
        # 记录误差
        eval_loss += loss.data[0]
        # 记录准确率
        _, pred = out.max(1)
        # num_correct = (pred == label).sum().data[0]
        num_correct = (pred == label).sum()
        # acc = num_correct // im.shape[0]
        eval_acc += num_correct.item()
        # eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_set), eval_acc / len(test_set)))


# net = nn.Sequential(
#     nn.Linear(784, 400), #  因为28*28=748
#     nn.ReLU(),
#     nn.Linear(400, 200),
#     nn.ReLU(),
#     nn.Linear(200, 100),
#     nn.ReLU(),
#     nn.Linear(100, 10) #  最后输出10个分类
# )
# #
# import torch
# import torchvision
# import numpy as np
# from torch import nn
# from torch.nn import init
# from torch.autograd import Variable
# from torch.utils import data
#
# EPOCH = 20
# BATCH_SIZE = 64
# LR = 1e-4
#
# # mnist download,transform NHWC=>NCHW and 0,255=>0,1
# train_data=torchvision.datasets.MNIST(root='/mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# # pytorch's dataset loader
# train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# # test data
# test_data = torchvision.datasets.MNIST(root='/mnist', train=False)
# # test Variable  need transform  gpu
# test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)).cuda()/255
# test_y = test_data.test_labels.cuda()
# # create model
#
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1=nn.Sequential(
#         nn.Conv2d(in_channels=1,out_channels=16,kernel_size=4,stride=1,padding=2),
#         nn.MaxPool2d(kernel_size=2,stride=2))
#         self.conv2=nn.Sequential(nn.Conv2d(16,32,4,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
#         self.out=nn.Linear(32*7*7,10)
#         # init
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         per_out = []
#         x = self.conv1(x)
#         per_out.append(x)
#         x = self.conv2(x)
#         per_out.append(x)
#         x = x.view(x.size(0),-1)
#         output=self.out(x)
#         # can output middle layer's features
#         return output, per_out
#
# cnn = CNN().cuda()  # gpu
#
# # set different layer's learning rate: [conv1 conv2] lr*10 ; [out]  lr
# def get_10x_lr_params(net):
#     b = [net.conv1,net.conv2]
#     for i in b:
#         for j in i.modules():
#             for k in j.parameters():
#                 yield k
#                 # generator
# # fine-tune
# new_params = cnn.state_dict()
# pretrain_dict = torch.load('/model/model.pth')
# pretrain_dict = {k : v for k, v in pretrain_dict.items() if k in new_params and v.size() == new_params[k].size()}  # dict gennerator
# new_params.update(pretrain_dict)
# cnn.load_state_dict(new_params)
#
# cnn.train()  # if you want test ,just modify cnn.eval()
#
# # update lr
# def lr_poly(base_lr,iters,max_iter,power):
#     return base_lr*((1-float(iters)/max_iter)**power)
# def adjust_lr(optimizer,base_lr,iters,max_iter,power):
#     lr = lr_poly(base_lr,iters,max_iter,power)
#     optimizer.param_groups[0]['lr'] = lr  # first param iterator
#     if len(optimizer.param_groups) > 1:
#         optimizer.param_groups[1]['lr']=lr*10
#
# Params = get_10x_lr_params(cnn)
# # optimizer             first params  lr=LR Internal overlapping external lr; second params
# optimizer = torch.optim.Adam([{'params':cnn.out.parameters()},{'params':Params,'lr':LR*10}],lr=LR)
# # loss function
# loss_func = nn.CrossEntropyLoss().cuda()
#
# iters = 0
# for epoch in range(EPOCH):
#     i_iter = train_data.train_data.shape[0]//BATCH_SIZE
#     for step, (x, y) in enumerate(train_loader):
#         optimizer.zero_grad()  # clear gradient
#         adjust_lr(optimizer, LR, iters, EPOCH*i_iter, 0.9)  # update lr
#         iters += 1
#         b_x = Variable(x).cuda()# if channel==1 auto add c=1
#         b_y = Variable(y).cuda()
#         # print(cnn.state_dict()['conv1.0.weight'])
#         output = cnn(b_x)[0]
#         loss = loss_func(output,b_y)# Variable need to get .data
#         loss.backward() # backward loss
#         optimizer.step() # compute per gradient
#
#         if step % 50 == 0:
#             test_output = cnn(test_x)[0]
#             pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
#             '''
#             why data ,because Variable .data to Tensor;and cuda() not to numpy() ,must to cpu
#             and to numpy and .float compute decimal
#             '''
#             accuracy = torch.sum(pred_y == test_y).data.float()/test_y.size(0)
#             print('EPOCH: ',epoch,'| train_loss:%.4f'% loss.data[0],'| test accuracy:%.2f'%accuracy)
#             #                                           loss.data.cpu().numpy().item() get one value
#             torch.save(cnn.state_dict(),'./model/model.pth')
# # test phase
# test_output = cnn(test_x[:13])[0]
# pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
# print(pred_y)
# print(test_y[:13])
