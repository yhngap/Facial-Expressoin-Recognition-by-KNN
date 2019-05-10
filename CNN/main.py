from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import Model
from torchvision import datasets, transforms
import cv2
import face_recognition
import numpy as np
from model import *
from VGG import *
from resnet import *
from alexnet import *

emotion={0:"angry",1:"disgust",2:"fear",3:"happy",4:"sad",5:"surprise",6:"neutral"}
use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
if use_gpu:
    dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_trained_model():
    PATH="./model"
    exists = os.path.isfile(PATH)
    if exists:
        print("loaded old model!")
        return torch.load(PATH, map_location='cpu' if not use_gpu else 'cuda:0')
    print("loaded new model!")
    return Model()


def train_model(model, criterion, optimizer, num_epochs=1):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(42),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(42),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    }

    data_dir = r"./training"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                correctness=[1 if preds[i] == labels.data[i] else 0 for i in range(len(preds))]
                running_corrects += torch.sum(torch.cuda.FloatTensor(correctness))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

        print('Best test Acc: {:4f}'.format(best_acc))
        torch.save(model, 'model')


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

def video_recognition(model):
    mini_batch = []
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)

        for top, right, bottom, left in face_locations:
            face_image = frame[top:bottom, left:right]
            cv2.rectangle(frame, (left,top), (right,bottom), (0, 255, 0), 3)
            small_image=cv2.resize(face_image, (42, 42))
            gray_image=cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("gray small",gray_image)
            gray_image=np.array(gray_image)
            gray_image=np.reshape(gray_image,(1,42,42))
            mini_batch.clear()
            for i in range(32):
                mini_batch.append(gray_image)
            output=model(torch.FloatTensor(mini_batch))
            _, pred = torch.max(output.data, 1)
            pred=pred.tolist()
            pred=[emotion[pred[i]] for i in pred]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, most_frequent(pred),(left,top), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    test_resnet()
    test_alexnet()
    test_vgg()
    test_mymodel()
    print("please input the number you want to do:")
    user_input = input("1: train model, 2: real-time demo, 3: exit program ")
    if user_input == 1:
        model = Model()  # either get the pre-trained model or trained
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        train_model(model, criterion, optimizer, num_epochs=1)
    if user_input == 2:
        model.train(False)
        video_recognition(model)
    if user_input == 3:
        pass


if __name__ == '__main__':
    main()
