import scipy.misc
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
  
def feature_visual(features, label, channels = 16): 
    # print(features.shape)   
    for i in range(channels): 
        feature = features[:,i,:,:]
        feature=feature.data.numpy()
        feature= 1.0/(1+np.exp(-1*feature))
        feature=np.round(feature*255)
        scipy.misc.imsave('features/'+str(i)+'-'+str(label)+'.jpg', feature[0])

def show_heatmap(mask, label, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    plt.imshow(heatmap)
    plt.axis('off')
    plt.savefig(str(label)+"-5.jpg")

    # img = img.numpy()[0].transpose(1, 2, 0)
    # # print(img.shape)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # img = img[:, :, :].squeeze()
    # # print(np.max(img), np.min(img))
    # plt.imshow(img, cmap=plt.cm.gray_r)
    # plt.axis('off')
    # plt.savefig(str(label)+"img.jpg")

    # rgb = img.convert('RGB')
    # cam = rgb + heatmap
    # cam = cam / 2
    # plt.imshow(cam)
    # plt.axis('off')
    # plt.savefig(str(label)+"cam.jpg")
