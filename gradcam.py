import cv2
import numpy as np
import torch
from torch.autograd import Function
from model import Net
from torch import nn

def Grad_Cam(model, feature_module, img, index=None):
    feature = model.get_feature(img).data.numpy()[0] # 卷积层输出的特征
    # print("featue shape: ", feature.shape)
    output = model(img) # 输出结果

    if index == None:
        index = np.argmax(output.data.numpy())

    # 选择一个class作为可视化的目标
    choose_class = np.zeros((1, output.size()[-1]), dtype=np.float32)
    choose_class[0][index] = 1
    choose_class = torch.from_numpy(choose_class).requires_grad_(True)

    # 计算论文中的alpha_k^c
    class_loss = torch.sum(choose_class * output)
    model.zero_grad()
    class_loss.backward()

    feature_grad = model.get_feature_grad(img)[0].data.numpy()[0]
    # print(np.sum(feature_grad))
    # print("grad shape: ", feature_grad.shape)
    weights = np.mean(feature_grad, axis=(1, 2))
    # print(weights)

    # 计算不同通道的加权
    cam_pic = np.zeros(feature.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam_pic += w * feature[i, :, :]
    
    # 可视化结果
    cam_pic = np.maximum(cam_pic, 0)
    # cam_pic = np.minimum(cam_pic, 0)
    # print(cam_pic)
    # print("cam shape: ", cam_pic.shape)
    # print("image shape: ", img.shape)
    cam_pic = cv2.resize(cam_pic, img.shape[2:])
    cam_pic = (cam_pic - np.min(cam_pic)) / (np.max(cam_pic) - np.min(cam_pic))

    return cam_pic