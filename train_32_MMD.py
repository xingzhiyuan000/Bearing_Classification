import itertools

import torch
import torchvision
import os
import torch.nn.functional as F

from torch import nn
from torch.distributions import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
import time
from torch.optim import Adam, lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from nets.Great import Great
from nets.MMDLoss import MMDLoss
import yaml
from nets.densenet import densenet_bearing
from nets.ghostnet import ghostnet
from nets.mobilenet_v1 import mobilenet_v1
from nets.mobilenet_v2 import mobilenet_v2
from nets.mobilenet_v3 import mobilenet_v3
from nets.resnet import ResNet, Bottleneck
from loss_funcs.lmmd import LMMDLoss

with open('./Training_Config.yaml', 'r', encoding='utf-8') as file:
    yaml_data = yaml.load(file.read(), Loader=yaml.FullLoader)

def train(source, target, lamb):
    # print(yaml_data['save_epoch'])

    # 保存信息
    # output_list=[]
    # 定义方法
    # def forward_hook(module,data_input,data_output):
    #     output_list.append(data_output)

    #tensorboard使用方法：tensorboard --logdir "E:\Python\Fault Diagnosis\Classification\logs"
    #需要设置cuda的数据有: 数据，模型，损失函数

    save_epoch=yaml_data['save_epoch'] #模型保存迭代次数间隔-10次保存一次
    Resume = yaml_data['Resume'] #设置为True是继续之前的训练 False为从零开始
    path_checkpoint = yaml_data['path_checkpoint'] #模型路径

    #定义训练的设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #准备数据集
    #加载自制数据集

    root = source  # 【资源域】数据集-带标签
    root_target = target  # 【目标域】数据集-无标签

    #读取资源域数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root,0.2)
    #读取目标域数据
    target_train_images_path, target_train_images_label, target_val_images_path, target_val_images_label = read_split_data(root_target,0.2)

    data_transform = {
        "train": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        "val": torchvision.transforms.Compose([torchvision.transforms.ToTensor()])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    test_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])
    tgt_train_data_set = MyDataSet(images_path=target_train_images_path,
                               images_class=target_train_images_label,
                               transform=data_transform["train"])
    tgt_test_data_set = MyDataSet(images_path=target_val_images_path,
                               images_class=target_val_images_label,
                               transform=data_transform["train"])

    train_data_size=len(train_data_set)
    test_data_size=len(test_data_set)
    tgt_train_data_size=len(tgt_train_data_set) #目标域数据集长度
    tgt_test_data_size=len(tgt_test_data_set) #目标域数据集长度
    #加载数据集
    batch_size = yaml_data['batch_size']
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers'.format(nw))
    # drop_last=True 用来舍弃不足的末尾BatchSize
    train_dataloader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=test_data_set.collate_fn)
    tgt_train_dataloader = torch.utils.data.DataLoader(tgt_train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=tgt_train_data_set.collate_fn)
    tgt_test_dataloader = torch.utils.data.DataLoader(tgt_test_data_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=tgt_test_data_set.collate_fn)
    #plot_data_loader_image(train_dataloader)

    #------------------------------------------------------#
    #   创建模型
    # 定义注意力机制 0-无注意力 1-SE 2-CBAM 3-ECA 4-CA
    #------------------------------------------------------#
    if(yaml_data['backbone']=="densenet_bearing"):
        wang=densenet_bearing()
    elif(yaml_data['backbone']=="Great"):
        wang = Great()
    elif (yaml_data['backbone'] == "ghostnet"):
        wang = ghostnet()
    elif (yaml_data['backbone'] == "mobilenet_v1"):
        wang = mobilenet_v1()
    elif (yaml_data['backbone'] == "mobilenet_v2"):
        wang = mobilenet_v2()
    elif (yaml_data['backbone'] == "mobilenet_v3"):
        wang = mobilenet_v3()
    elif (yaml_data['backbone'] == "resnet"):
        wang = ResNet(Bottleneck, [3, 4, 6, 3])



    #对已训练好的模型进行微调
    if Resume:
        pre_weights=torch.load(path_checkpoint, map_location=torch.device('cuda'))
        torch.save(pre_weights.state_dict(), './tmp.pth') #将其存成/tmp/1.pth文件
        wang.load_state_dict(torch.load('./tmp.pth'))

    wang = wang.to(device)  # 将模型加载到cuda上训练
    # 注册hook
    # wang.fc1.register_forward_hook(forward_hook)

    #定义损失函数
    loss_fn=nn.CrossEntropyLoss()
    loss_fn=loss_fn.to(device) #将损失函数加载到cuda上训练
    if (yaml_data['transfer_loss'] == "mmd"):
        transfer_loss=MMDLoss().to(device) #MMD域适应损失
    elif (yaml_data['transfer_loss'] == "lmmd"):
        transfer_loss = LMMDLoss(num_class=int(yaml_data['num_class'])).to(device)  # MMD域适应损失


    # beta=yaml_data['beta'] #控制MMD正则项强度


    #定义优化器
    learing_rate=yaml_data['learing_rate'] #学习速率
    # optimizer=torch.optim.SGD(wang.parameters(),lr=learing_rate)
    optimizer = Adam(wang.parameters(), lr=float(learing_rate))  # 选用AdamOptimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #使用学习率指数连续衰减

    #设置训练网络的一些参数
    total_train_step=0 #记录训练的次数
    total_test_step=0 #记录测试的次数
    epoch=yaml_data['epoch'] #训练的轮数

    #添加tensorboard
    # writer=SummaryWriter("logs",flush_secs=5)
    test_accuracy=np.array([])
    target_test_accuracy=np.array([])
    model_name = np.array([])
    for i in range(epoch):
        print("---------第{}轮训练开始------------".format(i+1))
        total_train_loss=0 #训练集整体Loss
        total_clf_loss = 0  # 整体分类Loss
        total_transfer_loss = 0  # 整体迁移Loss
        #训练步骤开始
        wang.train() #会对归一化及dropout等有作用
        for data in train_dataloader:
            # output_list=[]
            imgs, targets=data #取图片数据
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs=wang(imgs) #放入网络训练
            source_feature = wang.featuremap.transpose(1, 0).cpu()
            # print(source_feature)
            # print(source_feature.shape)
            source_fc_data = source_feature.cpu().detach().numpy().T

            source_fc_data = torch.tensor(source_fc_data)
            # print(source_fc_data.shape)
            source_batch_size = targets.size()[0] #资源域输入Batch大小

            # print("[资源域]取到的Batch：", source_batch_size)
            target_imgs, _ = next(iter(tgt_train_dataloader)) #获取目标域下一次的迭代数据
            target_batch_size = target_imgs.size()[0]  # 资源域输入Batch大小
            # print("[目标域]取到的Batch：",target_batch_size)
            if(source_batch_size!=target_batch_size):
                break
            # target_imgs, _ = next(iter(tgt_train_dataloader))  # 直接取目标域下一个Batch的数据
            target_imgs = target_imgs.to(device)  # 将图片加载到cuda上训练
            target_outputs = wang(target_imgs)  # 放入网络预测
            target_feature = wang.featuremap.transpose(1, 0).cpu()

            # print(target_feature)
            # print(target_feature.shape)
            target_fc_data = target_feature.cpu().detach().numpy().T

            target_fc_data = torch.tensor(target_fc_data)
            # print(target_fc_data.shape)

            loss1 = loss_fn(outputs, targets)  # 用损失函数计算误差值-【分类损失】
            # print('loss1:', loss1)
            if (yaml_data['transfer_loss'] == "mmd"):
                loss2 = transfer_loss(source_fc_data, target_fc_data)
            elif (yaml_data['transfer_loss'] == "lmmd"):
                # source_label=targets.cpu().detach().numpy()
                # print(targets.size()[0])
                target_logits=torch.nn.functional.softmax(target_outputs, dim=1)
                loss2 = transfer_loss(source_fc_data.to(device), target_fc_data.to(device),targets,target_logits)
            # print('loss2:', "%2.20f"%loss2)
            loss = loss1 + lamb * loss2
            #优化器调优
            optimizer.zero_grad() #清零梯度
            loss.backward() #反向传播
            optimizer.step()
            total_train_loss = total_train_loss + loss.item()
            total_clf_loss = total_clf_loss + loss1.item()
            total_transfer_loss = total_transfer_loss + loss2.item()
            total_train_step=total_train_step+1
            if total_train_step%50==0:
                print("总训练批次: {},【整体】Loss: {}, 【分类】Loss: {}, 【迁移】Loss: {}".format(total_train_step,loss.item(),
                                                                                  loss1.item(), loss2.item()))

                # writer.add_scalar("train_loss",loss.item(),global_step=total_train_step)

        if (i+1) % 2 == 0:
            scheduler.step() #每2次调整一次学习率
        current_learn_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print("当前学习率：", current_learn_rate, "---------------------------")
        print("第{}训练后的【训练集-整体】Loss为: {}".format(i + 1, total_train_loss))
        print("第{}训练后的【训练集-分类】Loss为: {}".format(i + 1, total_clf_loss))
        print("第{}训练后的【训练集-迁移】Loss为: {}".format(i + 1, total_transfer_loss))
        #一轮训练后，进行测试
        wang.eval()
        total_test_loss=0 #总体loss
        total_correct_num=0 #总体的正确个数
        transfer_total_test_loss = 0  # 总体loss
        transfer_total_correct_num = 0  # 总体的正确个数

        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets=data
                imgs = imgs.to(device)  # 将图片加载到cuda上训练
                targets = targets.to(device)  # 加载到cuda上训练
                outputs=wang(imgs)
                loss_clf=loss_fn(outputs,targets) #单个数据的loss
                total_test_loss=total_test_loss+loss_clf
                correct_num=(outputs.argmax(1)==targets).sum() #1:表示横向取最大值所在项
                total_correct_num=total_correct_num+correct_num #计算预测正确的总数
        test_accuracy=np.append(test_accuracy,(total_correct_num/test_data_size).cpu()) #保存每次迭代的测试准确率
        print("第{}训练后的【测试集-资源域】总体Loss为: {}".format(i+1,total_test_loss))
        print("第{}训练后的【测试集-资源域】总体正确率为: {}".format(i+1,total_correct_num/test_data_size))
        # writer.add_scalar("test_loss",total_test_loss, total_test_step) #添加测试loss到tensorboard中
        # writer.add_scalar("test_accuracy",total_correct_num/test_data_size,total_test_step) #添加测试数据集准确率到tensorboard中
        # total_test_step=total_test_step+1

        #--------------------跨域测试----------------------#
        with torch.no_grad():
            for data in tgt_test_dataloader:
                imgs, targets=data
                imgs = imgs.to(device)  # 将图片加载到cuda上训练
                targets = targets.to(device)  # 加载到cuda上训练
                outputs=wang(imgs)
                loss_transfer=loss_fn(outputs,targets) #单个数据的loss
                transfer_total_test_loss=transfer_total_test_loss+loss_transfer
                correct_num=(outputs.argmax(1)==targets).sum() #1:表示横向取最大值所在项
                transfer_total_correct_num=transfer_total_correct_num+correct_num #计算预测正确的总数
        target_test_accuracy=np.append(target_test_accuracy,(transfer_total_correct_num/tgt_test_data_size).cpu()) #保存每次迭代的测试准确率
        print("第{}训练后的【测试集-目标域】总体Loss为: {}".format(i+1,transfer_total_test_loss))
        print("第{}训练后的【测试集-目标域】总体正确率为: {}".format(i+1,transfer_total_correct_num/tgt_test_data_size))
        # writer.add_scalar("test_loss",total_test_loss, total_test_step) #添加测试loss到tensorboard中

        time_str=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        save_path = './models/'
        filepath = os.path.join(save_path, "wang_{}_{}.pth".format(time_str,i+1))
        model_name=np.append(model_name,filepath)
        if (i+1) % save_epoch == 0:
            torch.save(wang,filepath) #保存训练好的模型

    index=np.argmax(target_test_accuracy)
    str1 = '---------训练信息：【{}---{}】,正则强度: {}---------'.format(source[-1], target[-1], lamb)
    str2='第{}次迭代【测试集-资源域】准确率:{}'.format(np.argmax(target_test_accuracy)+1,test_accuracy[index])
    str3='第{}次迭代【测试集-目标域】最大准确率:{}'.format(np.argmax(target_test_accuracy) + 1, np.max(target_test_accuracy))
    str4='第{}次迭代对应的模型名称:{}'.format(np.argmax(target_test_accuracy) + 1,model_name[index])
    print(str1)
    print(str2)
    print(str3)
    print(str4)

    with open('./logs/result.txt', 'a') as file:
        file.write(str1 + '\n')
        file.write(str2 + '\n')
        file.write(str3 + '\n')
        file.write(str4 + '\n')


    # writer.close() #关闭tensorboard

    # print('第{}次迭代产生Accuracy最大值:{}'.format(np.argmax(test_accuracy),np.max(test_accuracy)))
    #
    # fig, ax=plt.subplots()
    # x=np.arange(1,epoch+1,1)
    # ax.plot(x,test_accuracy)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy/%')
    #
    # plt.show()
    # print(test_accuracy)

if __name__ == "__main__":
    for i in yaml_data['groups']:
        for j in yaml_data['groups']:
            if i != j:  # 排除重复组合
                source = yaml_data['root']+str(i)  # 【资源域】数据集-带标签
                target = yaml_data['root']+str(j)  # 【目标域】数据集-无标签

                for k in np.linspace(yaml_data['lambda_range'][0], yaml_data['lambda_range'][1], yaml_data['lambda_range'][2]):

                    # combination = (i, j)
                    # print(yaml_data['lambda_range'][0])
                    # print(source,'-',target)
                    # print(k)
                    '''
                    训练模型
                    '''
                    train(source,target,k)
