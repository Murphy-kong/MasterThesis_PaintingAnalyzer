import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import torchvision
import torch.nn.functional as F 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pathlib
from pathlib import Path
import pandas as pd
import PIL
from PIL import Image
from skimage import io
from skimage.color import gray2rgb
from sklearn import preprocessing
from torchmetrics.classification import MulticlassF1Score
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Tuple
import time
from sklearn.metrics import f1_score 

Image.MAX_IMAGE_PIXELS = None

mean = np.array([0.485, 0.456, 0.406])
std = np.array( [0.229, 0.224, 0.225])


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Applying Transforms to the Data
data_transforms = { 
     'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class customdataset(Dataset):
    def __init__(self, csv_file, csv_file2, root_dir, transform, n =None):
        if n == None:
            df = pd.read_csv(csv_file)
            df2 = pd.read_csv(csv_file2)
        else:
            df = pd.read_csv(csv_file, nrows=n)
            df2 = pd.read_csv(csv_file2, nrows=n)
        frames = [df, df2]
        df = pd.concat(frames)
        unsorted_labels = {x: df[x].unique() for x in ['artist','style','genre']}
        self.labels = {x: np.sort(unsorted_labels[x]) for x in ['artist','style','genre']}
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')

        image_artist = self.annotations.iloc[index,1]
        image_style = self.annotations.iloc[index,2]
        image_genre = self.annotations.iloc[index,3]

        #test = le.transform(list(le.classes_))
        #le fit anpassen an die unique df col
        le = preprocessing.LabelEncoder()
        le.fit(self.labels['style'])
        image_style = le.transform([image_style])
        le.fit(self.labels['artist'])
        image_artist = le.transform([image_artist])
        le.fit(self.labels['genre'])
        image_genre = le.transform([image_genre])
    
        if self.transform:
            #image =  np.array(image)
            #size = image.shape
            #print(size)
            #if(size[2] == 1):
                #image = gray2rgb(image)
            image = self.transform(image)
        
        return(image,image_artist,image_style,image_genre)

csv_path_train = 'groundtruth_multiloss_train_header.csv' 
csv_path_test = 'groundtruth_multiloss_test_header.csv' 


# Hyper-parameters 
num_epochs = 2
learning_rate = 0.0001
batch_size = 32
v_and_t_split = 1000
# Dictionary for Dataset
image_datasets = {}

image_datasets['train'] = customdataset(csv_file =csv_path_train,csv_file2 =csv_path_test, root_dir="original/images", transform=data_transforms['train'] )
#image_datasets['test'] = customdataset(csv_file =csv_path_test, root_dir="original\images", transform=data_transforms['test'], n=2000  )

proportions = [.70, .30]
lengths = [int(p * len(image_datasets['train'])) for p in proportions]
lengths[-1] = len(image_datasets['train']) - sum(lengths[:-1])
image_datasets['train'],image_datasets['test']  = torch.utils.data.random_split(image_datasets['train'] , lengths)

#[14997, 14998] <--- muss als liste übergeben werden mit integer werten für die datensatzlänge
dataset_sizes = {x: len(image_datasets [x]) for x in ['train','test']}



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=8, drop_last=True)
              for x in ['train','test']}

# 80% train, 20% test

df = pd.read_csv(csv_path_train)

class_length = {}
class_length = {x:len(df[x].unique())
            for x in ['artist', 'style','genre']
}



class_names = {}
class_names = {x:df[x].unique()
            for x in ['artist', 'style','genre']
}

n_total_steps = len(dataloaders['train'])


#model
class Resnet50_multiTaskNet(nn.Module):
    def __init__(self):
        super(Resnet50_multiTaskNet, self).__init__()
        
        #self.model =  models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        #self.model =  models.resnet50(weights=None)
        self.model =  models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        #self.model = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model.parameters():
           # if param.requires_grad == True:
           #     print(param)
            param.requires_grad = True

        self.fc_artist = nn.Linear(2048, class_length ['artist']).to(device)
        self.fc_style = nn.Linear(2048, class_length ['style']).to(device)
        self.fc_genre = nn.Linear(2048, class_length ['genre']).to(device)


    

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x_artist = self.fc_artist(x)
        x_style = self.fc_style(x)
        x_genre = self.fc_genre(x)
        return x_artist, x_style, x_genre
    
    
#multitaskloss
class MultiTaskLoss(nn.Module):
    def __init__(self, model, loss_fn, eta) -> None:
        super(MultiTaskLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eta = nn.Parameter(torch.Tensor(eta))

    def forward(self, input, targets) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input)
        loss = [l(o,y) for l, o, y in zip(self.loss_fn, outputs, targets)]
        total_loss = torch.stack(loss) * torch.exp(-self.eta) + self.eta
        return loss, total_loss.sum(), outputs  # omit 1/2




#model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)

#for param in model.parameters():
    #param.requires_grad = False   

#model.fc = nn.Sequential(
               #nn.Linear(2048, class_length['artist'])).to(device)

model = Resnet50_multiTaskNet().to(device)        
criterion = nn.CrossEntropyLoss().to(device)

def loss_fn1(x, cls):
    return 2 * criterion(x, cls)
def loss_fn2(x, cls):
    return 2 * criterion(x, cls)

#mtl = MultiTaskLoss(model=Resnet50_multiTaskNet(),
#                    loss_fn=[loss_fn1, loss_fn2],
#                    eta=[1.0, 1.0]).to(device)  


optimizer = optim.SGD(model.fc_artist.parameters(), lr=0.001, momentum=0.9, weight_decay =  0.00001)
#optimizer2 = optim.Adam(model.fc_style.parameters())
#optimizer3 = optim.Adam(model.fc_genre.parameters())


#optimizer2 = optim.SGD([
        #{"params":model.fc_genre.parameters(),"lr": 0.001,  "momentum":0.9},
        #{"params":model.fc_style.parameters(), "lr": 0.001,"momentum":0.9}])
#optimizer2 = optim.SGD(model.fc_style.parameters(), lr=0.01,  momentum=0.9, weight_decay =  0.00001)
optimizer2 = optim.SGD(model.fc_style.parameters(), lr=0.001,  momentum=0.9, weight_decay =  0.00001)
optimizer3 = optim.SGD(model.fc_genre.parameters(), lr=0.001,  momentum=0.9, weight_decay =  0.00001)
optimizer4 = optim.SGD(model.model.parameters(), lr=0.001,  momentum=0.9, weight_decay =  0.0001)
#der wurde auskommentiertoptimizer4 = optim.SGD(model.model.parameters(), lr=0.00001,  momentum=0.9, weight_decay =  0.00001)
print(n_total_steps/batch_size)
#optimizer2 = optim.Adam([{"params": model.fc_style.parameters()},
#                       {"params": model.fc_genre.parameters()}])
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.1)
exp_lr_scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.1)
exp_lr_scheduler4 = lr_scheduler.StepLR(optimizer4, step_size=10, gamma=0.1)
#model.load_state_dict(torch.load("cnn_transfer_learning/modelsavepoints/multitask_resnet_nofreezedlayers_moredataaugmentation_01_0001_0001_weight_null_epoch75_2.pth"))

#model.load_state_dict(torch.load("cnn_transfer_learning/modelsavepoints/multitask_resnet_nofreezedlayers_moredataaugmentation_01_0001_0001_weight_null_epoch45.pth"))
#model.load_state_dict(torch.load("cnn_transfer_learning/modelsavepoints/resnet50_pretrained_cnn_model_45epoch_ohnepretrained_best_Acc.pth"))
model.load_state_dict(torch.load("cnn_transfer_learning/modelsavepoints/resnet50_pretrained_cnn_model_45epoch_warmupV1_best_Acc.pth"))
#DAS TRAINING WAR AUF LR 0.001
#lr war auf 0.001 bei style und genre, bei artist auf 0.1 und step size auf 30
#train Method
def train_model(model, criterion, optimizer, optimizer2, optimizer3,optimizer4, exp_lr_scheduler, exp_lr_scheduler2, exp_lr_scheduler3,exp_lr_scheduler4,num_epochs=3 ):
    since = time.time()
    test_accu = []
    test_accu_artist = []
    test_accu_style = []
    test_accu_genre = []
    test_losses = []
    test_losses_artist = []
    test_losses_style = []
    test_losses_genre = []

    train_accu = [] 
    train_accu_artist = []
    train_accu_style = []
    train_accu_genre = []
    train_losses = []
    train_losses_artist = []
    train_losses_style = []
    train_losses_genre = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_artist = 0
            running_corrects_style = 0
            running_corrects_genre = 0
            running_loss_artist = 0
            running_loss_style = 0
            running_loss_genre = 0
            i = 0
            t = 0
    
            for inputs,image_artist,image_style,image_genre in dataloaders[phase]:
                i = i + 1
                inputs = inputs.to(device)
                image_artist = image_artist.to(device)
                image_style = image_style.to(device)
                image_genre = image_genre.to(device)
                image_artist=  torch.squeeze(image_artist).long() 
                image_style=  torch.squeeze(image_style).long()
                image_genre=  torch.squeeze(image_genre).long()
                outputs = model(inputs)
                #loss, total_loss, outputs = model(inputs, [image_artist, image_style])   
                loss_artist = criterion(outputs[0], image_artist)
                loss_style = criterion(outputs[1], image_style)
                loss_genre = criterion(outputs[2], image_genre)

                loss = (loss_artist + loss_style + loss_genre) / 3


                if phase == 'train':
                    optimizer.zero_grad()
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()
                    #if epoch < 20:
                    optimizer4.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer2.step()
                    optimizer3.step()
                    #if epoch < 20:
                    optimizer4.step()


                _, preds_artist = torch.max(outputs[0], 1)
                _, preds_style = torch.max(outputs[1], 1)  
                _, preds_genre = torch.max(outputs[2], 1)    
                running_loss += loss.detach() * inputs.size(0)
                running_loss_artist += loss_artist.detach() * inputs.size(0)
                running_loss_style +=  loss_style.detach() * inputs.size(0)
                running_loss_genre += loss_genre.detach() * inputs.size(0)
                running_corrects_artist += torch.sum(preds_artist == image_artist.data)
                running_corrects_style += torch.sum(preds_style == image_style.data)
                running_corrects_genre += torch.sum(preds_genre == image_genre.data)
               # current_Acc_artist = running_corrects_artist.float() / len(image_datasets[phase])
               # current_Acc_style = running_corrects_style.float() / len(image_datasets[phase])
               # current_Acc_genre = running_corrects_genre.float() / len(image_datasets[phase])
               # f1_score_artist_micro = f1_score(image_artist.data.cpu(), preds_artist.cpu(), average= "micro")
               # f1_score_artist_macro = f1_score(image_artist.data.cpu(), preds_artist.cpu(), average= "macro")
               # f1_score_artist_weighted = f1_score(image_artist.data.cpu(), preds_artist.cpu(), average= "weighted")
               # f1_score_style = f1_score(image_style.data.cpu(), preds_style.cpu(), average= "micro")
               # f1_score_genre = f1_score(image_genre.data.cpu(), preds_genre.cpu(), average= "micro")
                #if(i % 100 == 0):
                    #t = t + 1
                    #print('{}. {} acc: artist: {:.4f}  style: {:.4f} genre: {:.4f}'.format(t,phase,current_Acc_artist.item(),current_Acc_style.item(), current_Acc_genre.item() ))
                    
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc_artist = running_corrects_artist.float() / len(image_datasets[phase])
            epoch_acc_style = running_corrects_style.float() / len(image_datasets[phase])
            epoch_acc_genre = running_corrects_genre.float() / len(image_datasets[phase])
            epoch_acc = (epoch_acc_artist + epoch_acc_genre + epoch_acc_style) / 3
            running_loss_artist = running_loss_artist / len(image_datasets[phase])
            running_loss_style =  running_loss_style / len(image_datasets[phase])
            running_loss_genre = running_loss_genre / len(image_datasets[phase])
            if phase == 'test':
                test_accu.append(epoch_acc.item())
                test_accu_artist.append(epoch_acc_artist.item())
                test_accu_style.append(epoch_acc_style.item())
                test_accu_genre.append(epoch_acc_genre.item())
                test_losses.append(epoch_loss.item())
                test_losses_artist.append(running_loss_artist.item())
                test_losses_style.append(running_loss_style.item())
                test_losses_genre.append(running_loss_genre.item())
            if phase == 'train':
                train_accu.append(epoch_acc.item())
                train_accu_artist.append(epoch_acc_artist.item())
                train_accu_style.append(epoch_acc_style.item())
                train_accu_genre.append(epoch_acc_genre.item())
                train_losses.append(epoch_loss.item())
                train_losses_artist.append(running_loss_artist.item())
                train_losses_style.append(running_loss_style.item())
                train_losses_genre.append(running_loss_genre.item())



            exp_lr_scheduler.step()
            exp_lr_scheduler2.step()
            exp_lr_scheduler3.step()
            #if epoch < 30:
            #exp_lr_scheduler4.step()

            print('\n {} acc: {:.4f},acc: artist: {:.4f}  style: {:.4f} genre: {:.4f} \n'.format(phase,
                                                        epoch_acc.item(),
                                                        epoch_acc_artist.item(),
                                                        epoch_acc_style.item(),
                                                        epoch_acc_genre.item()
                                                        ))
            print('\n {} loss: {:.4f},loss: artist: {:.4f}  style: {:.4f} genre: {:.4f} \n'.format(phase,
                                                        epoch_loss.item(),
                                                        running_loss_artist.item(),
                                                        running_loss_style.item(),
                                                        running_loss_genre.item()
                                                        ))
            #print('\n {} loss: {:.4f},acc: artist micro: {:.4f} artist macro: {:.4f}  artist weighted: {:.4f}   style: {:.4f} genre: {:.4f} \n'.format(phase,
            #                                            epoch_loss.item(),
            #                                            f1_score_artist_micro.item(),
            #                                            f1_score_artist_macro.item(),
            #                                            f1_score_artist_weighted.item(),
            #                                            f1_score_style.item(),
            #                                            f1_score_genre.item()
            #                                            ))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    save_path = Path("cnn_transfer_learning/modelsavepoints/resnet50_pretrained_cnn_model_45epoch_warmupV1_best_Acc.pth")
    torch.save(model.state_dict(), save_path)
    #torch.save({
    #        'epoch': 20,
    #        'model_state_dict': model.state_dict(),
    #        'optimizer_state_dict': optimizer.state_dict(),
    #        'optimizer2_state_dict': optimizer2.state_dict(),
    #        'optimizer3_state_dict': optimizer3.state_dict(),
    #        }, save_path)



    return model, test_accu,test_accu_artist,test_accu_style,test_accu_genre,test_losses,test_losses_artist,test_losses_style,test_losses_genre,\
    train_accu, train_accu_artist, train_accu_style ,train_accu_genre ,train_losses ,train_losses_artist ,train_losses_style ,train_losses_genre

model, test_accu,test_accu_artist,test_accu_style,test_accu_genre,test_losses,test_losses_artist,test_losses_style,test_losses_genre,\
train_accu, train_accu_artist, train_accu_style ,train_accu_genre ,train_losses ,train_losses_artist ,train_losses_style ,train_losses_genre \
= train_model(model, criterion, optimizer,optimizer2,optimizer3,optimizer4, 
exp_lr_scheduler=  exp_lr_scheduler,exp_lr_scheduler2= exp_lr_scheduler2,exp_lr_scheduler3= exp_lr_scheduler3,exp_lr_scheduler4=exp_lr_scheduler4, num_epochs=20)
print(test_accu)
print(test_losses)
#plot accuracy
plt.clf()
plt.plot(train_accu)
plt.plot(test_accu)
plt.xlabel('epoch')
plt.ylabel('accuracy')
lgd = plt.legend(['Train','Test'],  bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Train vs Test Accuracy')
plt.savefig('output_acc.png', bbox_inches='tight')

#plot loss
plt.clf()
plt.plot(train_losses)
plt.plot(test_losses)
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train','Test'],  bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Train vs Test Losses')
plt.savefig('output_loss.png', bbox_inches='tight')


#plot accuracy artist, genre , style
plt.clf()
plt.plot(train_accu_artist)
plt.plot(train_accu_style)
plt.plot(train_accu_genre)
plt.plot(test_accu_artist)
plt.plot(test_accu_style)
plt.plot(test_accu_genre)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train_artist','Train_style', 'Train_style', 'Test_artist', 'Test_style', 'Test_genre'],  bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Train vs Test Accuracy')
plt.savefig('output_acc_all.png', bbox_inches='tight')

#plot loss artist, genre , style
plt.clf()
plt.plot(train_losses_artist)
plt.plot(train_losses_style)
plt.plot(train_losses_genre)
plt.plot(test_losses_artist)
plt.plot(test_losses_style)
plt.plot(test_losses_genre)
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train_artist','Train_style', 'Train_style', 'Test_artist', 'Test_style', 'Test_genre'],  bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Train vs Test Losses')
plt.savefig('output_loss_all.png', bbox_inches='tight')

