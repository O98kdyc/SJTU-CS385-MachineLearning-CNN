import torch
from model import Net
from datamanager import obtain_loader
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_func = nn.MSELoss(reduction='mean')

filename1 = 'stu/stuloss.txt'
filename2 = 'stu/stutrainacc.txt'
filename3 = 'stu/stutestacc.txt'
filename4 = 'stu/labelloss.txt'
filename5 = 'stu/labeltrainacc.txt'
filename6 = 'stu/labeltestacc.txt'

def one_hot(lable, depth=10):
    out=torch.zeros(lable.size(0),depth)
    idx=torch.LongTensor(lable).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

def train(model, data_loader, optimizer, print_every=10, loss_func=nn.MSELoss(reduction='mean')):
    count = 0
    for _, (images, labels) in enumerate(data_loader):
        x, labels = Variable(images), Variable(labels)
        labels = one_hot(labels)
        out = model(x)
        optimizer.zero_grad()
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()

        #print loss
        count = count + 1
        if(count % print_every == 0):
            print("training loss: ", loss.data.item())

    return model

def test(model, data_loader, flag=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
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

def train_teacher_net(flag=False, epoch=2):
    batch_size = 500
    train_loader, test_loader = obtain_loader(batch_size, batch_size)
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if flag:
        optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    print("Teacher Training")
    for _ in range(epoch):
        model = train(model, train_loader, optimizer, loss_func=loss_func)
        train_acc = test(model, train_loader)
        print('Train Accuracy: {} %'.format(train_acc))
        test_acc = test(model, test_loader)
        print('Test Accuracy: {} %'.format(test_acc))   
    print("Teacher Training Finished")
    return model

def compare_training(model_from_labels, model_from_teacher, 
                        teacher_net, data_loader, optimizer_labels, optimizer_student, print_every=10, test_loader=None):
    count = 0
    for _, (images, labels) in enumerate(data_loader):
        x, labels = Variable(images), one_hot(labels)
        y = teacher_net(x)
        y = Variable(y)

        out = model_from_teacher(x)
        optimizer_student.zero_grad()
        loss_student = loss_func(out, y)
        loss_student.backward()
        optimizer_student.step()

        loss = loss_func(out, labels)

        out = model_from_labels(x)
        optimizer_labels.zero_grad()
        loss_labels = loss_func(out, labels)
        loss_labels.backward()
        optimizer_labels.step()

        #print loss
        count = count + 1
        if(count % print_every == 0):
            print("labels training loss: ", loss_labels.data.item())
            # print("teacher training loss: ", loss_student.data.item())
            print("teacher training label loss: ", loss.item())

            with open(filename1, 'a') as r:
                r.write(str(loss.item()) +"\n")
            r.close()

            train_acc = test(model_from_teacher,data_loader, flag=True)
            with open(filename2, 'a') as r:
                r.write(str(train_acc) +"\n")
            r.close()  

            test_acc = test(model_from_teacher, test_loader)
            with open(filename3, 'a') as r:
                r.write(str(test_acc) +"\n")
            r.close()        

            with open(filename4, 'a') as r:
                r.write(str(loss_labels.data.item()) +"\n")
            r.close()

            train_acc = test(model_from_labels,data_loader, flag=True)
            with open(filename5, 'a') as r:
                r.write(str(train_acc) +"\n")
            r.close()  

            test_acc = test(model_from_labels, test_loader)
            with open(filename6, 'a') as r:
                r.write(str(test_acc) +"\n")
            r.close()          


    return model_from_labels, model_from_teacher
    
# if __name__ == '__main__':
#     teacher = train_teacher_net()
#     model_labels, model_student = Net(), Net()
#     epoch = 10
#     batch_size = 500
#     train_loader, test_loader = obtain_loader(batch_size, batch_size)
#     optimizer_labels = optim.SGD(model_labels.parameters(), lr=0.5, momentum=0.5)
#     optimizer_student = optim.SGD(model_student.parameters(), lr=0.5, momentum=0.5)
#     # optimizer_labels = optim.Adam(model_labels.parameters(), lr=0.01, weight_decay=5e-4)
#     # optimizer_student = optim.Adam(model_student.parameters(), lr=0.01, weight_decay=5e-4)
#     print("Comparing")
#     for _ in range(epoch):
#         model_labels, model_student = compare_training(model_labels, model_student,
#                                         teacher, train_loader, optimizer_labels, optimizer_student, test_loader=test_loader)
#         train_acc = test(model_labels, train_loader)
#         print("Label Training Evaluation")
#         print('Train Accuracy: {} %'.format(train_acc))
#         test_acc = test(model_labels, test_loader)
#         print('Test Accuracy: {} %'.format(test_acc))

#         train_acc = test(model_student, train_loader)
#         print("Student Training Evaluation")
#         print('Train Accuracy: {} %'.format(train_acc))
#         test_acc = test(model_student, test_loader)
#         print('Test Accuracy: {} %'.format(test_acc))

#     print("Finished")