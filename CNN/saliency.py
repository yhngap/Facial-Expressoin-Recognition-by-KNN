from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
from model import Model
from torchvision import datasets, transforms
import cv2
import face_recognition
import numpy as np
from main import *
import matplotlib.pyplot as plt

def compute_saliency_maps(X, y, model):
    # Make sure the model is in "test" mode
    model.train(False)
    model.eval()
    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    scores = model(X_var)
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
    scores.backward(torch.cuda.FloatTensor(torch.ones(32)))
    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze()
    return saliency
def show_saliency_maps(X, y,model):
    # Convert X and y from numpy arrays to Torch Tensors

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X,y, model)
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    X=X.cpu().numpy()
    X=X.squeeze()
    saliency = saliency.cpu().numpy()
    N = X.shape[0]

    l=[]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        l.append(emotion[y.cpu().numpy()[i]])
        print(emotion[y.cpu().numpy()[i]])
        plt.axis('off')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(6, 2)
    plt.title(l)
    plt.xlabel(l)
    plt.show()
