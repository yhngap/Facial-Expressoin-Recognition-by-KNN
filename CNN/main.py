from __future__ import print_function, division
import os
from torchvision import datasets, transforms
import cv2
import face_recognition
import numpy as np
from model import *
from VGG import *
from resnet import *
from alexnet import *
from saliency import *

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


def train_model(model, criterion, optimizer, num_of_epochs=1):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(42),
            transforms.ColorJitter(brightness=0.7),
            transforms.RandomRotation(30),
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
    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
                      'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
                      }
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True,
                                                        num_workers=4)}
    fer_sizes = {'train': len(image_datasets['train']),'test': len(image_datasets['test'])}
    best_acc = 0.0
    best_loss=1.0

    total_accuracy=[]
    total_loss=[]

    axis_epoch=np.arange(num_of_epochs,step=1)

    for epoch in range(num_of_epochs):
        print('Epoch ',epoch + 1)

        for stage in ['train', 'test']:
            if stage == 'train':
                model.train(True)
            else:
                model.train(False)

            intermediate_loss = 0.0
            intermediate_correctness = 0

            for data in dataloaders[stage]:
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if epoch==2:
                    pass
                    #show_saliency_maps(inputs,labels,model)

                if stage == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                intermediate_loss += loss.data
                correctness=[1 if preds[i] == labels.data[i] else 0 for i in range(len(preds))]
                intermediate_correctness += torch.sum(torch.cuda.FloatTensor(correctness))

            epoch_loss = intermediate_loss / fer_sizes[stage]
            epoch_acc = intermediate_correctness / fer_sizes[stage]

            print('Loss: ',epoch_loss,' Acc: ', epoch_acc)
            if stage == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if stage == 'test' and (epoch_loss < best_loss):
                best_loss = epoch_loss
        total_accuracy.append(best_acc.cpu().numpy())
        total_loss.append(best_loss.cpu().numpy())
        torch.save(model, 'model')
    total_accuracy=np.array(total_accuracy)
    total_loss=np.array(total_loss)

    print(total_accuracy)
    print(total_loss)

    plt.title('training accuracy')
    plt.plot(axis_epoch,total_accuracy)
    plt.legend(['Our Model'], loc='upper left')
    plt.show()




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
    print("please input the number you want to do:")
    user_input = input("1: train model, 2: real-time demo, 3: exit program ")
    if user_input == '1':
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

        model = Model()  # either get the pre-trained model or trained
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters())
        train_model(model, criterion, optimizer, num_of_epochs=25)

        model = Model()  # either get the pre-trained model or trained
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_model(model, criterion, optimizer, num_of_epochs=25)

        model = Model()  # either get the pre-trained model or trained
        model = model.to(device)
        optimizer = optim.RMSprop(model.parameters())
        train_model(model, criterion, optimizer, num_of_epochs=25)

        model = Model()  # either get the pre-trained model or trained
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters())
        train_model(model, criterion, optimizer, num_of_epochs=25)
        # model =VGG16()
        # optimizer = optim.Adam(model.parameters())
        # train_model(model, criterion, optimizer, num_of_epochs=25)
        # model =ResNet18()
        # optimizer = optim.Adam(model.parameters())
        # train_model(model, criterion, optimizer, num_of_epochs=25)
        # model = AlexNet()
        # optimizer = optim.Adam(model.parameters())
        # train_model(model, criterion, optimizer, num_of_epochs=25)
        # plt.legend(['Our model', 'VGG16', 'ResNet', 'AlexNet'], loc='upper left')
        # plt.show()
    if user_input == '2':
        model.train(False)
        video_recognition(model)
    if user_input == '3':
        pass


if __name__ == '__main__':
    pass
    main()
