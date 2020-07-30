import torch
from model import Net, Net1, Net2
from datamanager import obtain_loader
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
import numpy as np
from feature_visual import feature_visual, show_heatmap
from gradcam import Grad_Cam
import scipy.misc

use_cuda = torch.cuda.is_available()
filename1 = 'net2/trainloss.txt'
filename2 = 'net2/trainacc.txt'
filename3 = 'net2/testacc.txt'

def train(model, data_loader, optimizer, start, mid, end, print_every=10,
            loss_func=nn.CrossEntropyLoss(), test_loader=None):
    visual_img = get_first_img()
    if use_cuda:
        visual_img = visual_img.cuda()
    visual_img = Variable(visual_img)
    count = 0
    for _, (images, labels) in enumerate(data_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        x, labels = Variable(images), Variable(labels)
        out = model(x)
        optimizer.zero_grad()
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()

        #print loss
        count = count + 1
        if(count % print_every == 0):
            # print(loss.data.item())
            # with open(filename1, 'a') as r:
            #     r.write(str(loss.data.item()) +"\n")
            # r.close()

            train_acc = test(model,data_loader, flag=True)
            print(train_acc)

            if start and train_acc >= 82:
                start = False
                print("start")
                # print(x[0].unsqueeze(0).shape)
                start_feature = model.get_feature(visual_img)
                start_feature = start_feature.to(torch.device('cpu'))
                feature_visual(start_feature, "start", 6)

            if mid and train_acc >= 95:
                mid = False
                print("mid")
                mid_feature = model.get_feature(visual_img)
                mid_feature = mid_feature.to(torch.device('cpu'))
                feature_visual(mid_feature, "mid", 6)

            if end and train_acc >= 98:
                end = False
                print("end")
                end_feature = model.get_feature(visual_img)      
                end_feature = end_feature.to(torch.device('cpu'))
                feature_visual(end_feature, "end", 6)


            # with open(filename2, 'a') as r:
            #     r.write(str(train_acc) +"\n")
            # r.close()  

            # test_acc = test(model, test_loader)

            # with open(filename3, 'a') as r:
            #     r.write(str(test_acc) +"\n")
            # r.close()             

    return model, start, mid, end

def test(model, data_loader, flag=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            x, labels = Variable(images), Variable(labels)
            outputs = model(x)
            predicted = outputs.argmax(axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            if flag:
                if total >= 10000:
                    break
    acc = float(100 * correct / total)
    return acc

def get_first_img():
    _, grad_loader = obtain_loader(1, 1)
    for images, _ in grad_loader:
        return images



start, mid, end = True, True, True
epoch = 10 #5
batch_size = 100 #100
train_loader, test_loader = obtain_loader(batch_size, batch_size)
learning_rate = 0.3
model = Net()
if use_cuda:
    model = model.to(torch.device('cuda'))
# model.load()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for e in range(epoch):
    model, start, mid, end = train(model, train_loader, optimizer, start, mid, end, test_loader=test_loader)
#     model.save()
    # train_acc = test(model, train_loader)
    # print('Train Accuracy: {} %'.format(train_acc))
# test_acc = test(model, test_loader)
# print('Test Accuracy: {} %'.format(test_acc))

# print(model._modules.items())
# _, grad_loader = obtain_loader(1, 1)
# exist_label = []
# model = model.to(torch.device('cpu'))
# for images, labels in grad_loader:
#     # print(labels)
#     img = Variable(images)

#     # print(model._modules.items())
#     # print(model.conv2._modules.items())
#     index = None
#     cam = Grad_Cam(model, model, img, index)
#     pic = 1.0/(1+np.exp(-1*cam))
#     pic = np.round(pic*255)
#     # scipy.misc.imsave('gradcam'+'.jpg', pic)

#     if(int(labels) not in exist_label):
#         show_heatmap(pic, int(labels), images)
#         exist_label.append(int(labels))
    
#     if len(exist_label) >= 1:
#         break

# _, visual_loader = obtain_loader(1, 1)
# for images, labels in visual_loader:
#     print(labels)
#     img = Variable(images)
#     feature = model.get_feature(img)
#     feature_visual(feature, int(labels))
#     break