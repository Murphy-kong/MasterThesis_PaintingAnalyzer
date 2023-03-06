# import main Flask class and request object
from flask import Flask, request, jsonify
import inspect
from pprint import pprint
import json 
import torch
from torchvision.models import resnet50, ResNet50_Weights
import PIL
from PIL import Image
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from pathlib import Path
import requests
from azure.core.exceptions import (
    ResourceExistsError,
    ResourceNotFoundError
)

from azure.storage.fileshare import (
    ShareServiceClient,
    ShareClient,
    ShareDirectoryClient,
    ShareFileClient
)
from torchvision.models import vit_b_16, ViT_B_16_Weights
from flask_cors import CORS




class customdataset():
    def __init__(self, csv_file, csv_file2, root_dir, n =None):
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

    def __len__(self):
        return len(self.annotations)

    def transform(self, artist, style, genre):
        
        le = preprocessing.LabelEncoder()
        le.fit(self.labels['style'])
        image_style = le.inverse_transform([style.cpu()])
        le.fit(self.labels['artist'])
        image_artist = le.inverse_transform([artist.cpu()])
        le.fit(self.labels['genre'])
        image_genre = le.inverse_transform([genre.cpu()])

        
        return(image_artist,image_style,image_genre)

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

# Applying Transforms to the Data
data_transforms_vit = { 
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Device configuration
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#model
class Resnet50_multiTaskNet(nn.Module):
    def __init__(self):
        super(Resnet50_multiTaskNet, self).__init__()
        
        self.model =  models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        #self.model = nn.Sequential(*list(self.model.children())[:-1])
        #for param in self.model.parameters():
           # if param.requires_grad == True:
           #     print(param)
            # param.requires_grad = False 

        self.fc_artist = nn.Linear(2048, 1508).to(device)
        self.fc_style = nn.Linear(2048, 125).to(device)
        self.fc_genre = nn.Linear(2048, 41).to(device)


    

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


csv_path_train = 'flaskml/groundtruth_multiloss_train_header.csv' 
df = pd.read_csv(csv_path_train)

class_length = {}
class_length = {x:len(df[x].unique())
            for x in ['artist', 'style','genre']
}

class VGG_19_multiTaskNet(nn.Module):
    def __init__(self):
        super(VGG_19_multiTaskNet, self).__init__()
        
        #self.model =  models.vgg19_bn(weights= VGG19_BN_Weights.IMAGENET1K_V1)
        self.model =  models.vgg19_bn(weights= None)
        self.model.classifier = nn.Sequential(*[self.model.classifier[i] for i in range(4)])

        for param in self.model.parameters():
            param.requires_grad = True


        self.fc_artist = nn.Linear(4096, class_length ['artist']).to(device)
        self.fc_style = nn.Linear(4096, class_length ['style']).to(device)
        self.fc_genre = nn.Linear(4096, class_length ['genre']).to(device)


    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)

        x_artist = self.fc_artist(x)
        x_style = self.fc_style(x)
        x_genre = self.fc_genre(x)
        return x_artist, x_style, x_genre
        

class ViT_multiTaskNet(nn.Module):
    def __init__(self):
        super(ViT_multiTaskNet, self).__init__()
        
        self.model =  models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.encoder = feature_extractor[1]


        for param in feature_extractor.parameters():
            param.requires_grad = False 


        self.fc_artist = nn.Linear(768, class_length ['artist']).to(device)
        self.fc_style = nn.Linear(768, class_length ['style']).to(device)
        self.fc_genre = nn.Linear(768, class_length ['genre']).to(device)


 

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures

        x = x[:, 0]

        x_artist = self.fc_artist(x)
        x_style = self.fc_style(x)
        x_genre = self.fc_genre(x)
        return x_artist, x_style, x_genre


def predict(model, img_path, vit = False):
    csv_path_train = 'flaskml/groundtruth_multiloss_train_header.csv' 
    csv_path_test = 'flaskml/groundtruth_multiloss_test_header.csv' 
    image_datasets = customdataset(csv_file =csv_path_train,csv_file2 =csv_path_test, root_dir="original/images" )
    image = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')
    if(vit == True):
        image = data_transforms_vit['test'](image)
    else:
        image = data_transforms['test'](image)
    image = image.unsqueeze(0)
    image = image.to(device)
    model.eval()
    outputs = model(image)

    _, preds_artist = torch.max(outputs[0], 1)
    _, preds_style = torch.max(outputs[1], 1)
    _, preds_genre = torch.max(outputs[2], 1)
    print(preds_artist)
    transformed_values = image_datasets.transform(preds_artist, preds_style, preds_genre)
    print(transformed_values)
    m = nn.Softmax(dim=1)
    output_artist = m(outputs[0])
    output_style = m(outputs[1])
    output_genre = m(outputs[2])
    return output_artist[0][preds_artist][0].item(), output_style [0][preds_style][0].item(), output_genre [0][preds_genre][0].item() ,  transformed_values  

# create the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/query-example')
def query_example():
    return 'Query String Example'

@app.route('/form-example')
def form_example():
    path = Path('./DOWNLOADED_vgg19.pth')
    print(path.is_file())
    if path.is_file() == True:
        return 'VGG is there!!'
    else:
        return 'form example'

@app.route('/downloadpth_vgg19')
def downloadpth_vgg():
    # Verbindung zur Azure File Storage herstellen
    print("TTEST HIIEEER")
    connection_string = "DefaultEndpointsProtocol=https;AccountName=paintinganalyzerstorage;AccountKey=KXI7rZNbuALQYfQUhAlSvTxyHbVp639X28dFrqtYfNLGwAJM87HJ0yVlgjwb5JaKNWl7mpdLI0KR+AStEZd+RA==;EndpointSuffix=core.windows.net"
    #file_service = ShareServiceClient.from_connection_string(conn_str=connection_string)

    # Name des File Shares und des Dateipfades
    share_name = "vgg19"
    #file_path = "vgg19_pretrained_cnn_model_65epoch_best_Acc.pth"
    file_path = "test3.txt"
    dir_name = ""
    # Client für die Datei erstellen und herunterladen
    #share_client = ShareFileClient.from_connection_string(conn_str=connection_string, share_name=share_name, file_path=file_path)
    share_client = ShareClient.from_connection_string(connection_string, share_name)
    print("TEST")
    print(share_client)
    for item in list(share_client.list_directories_and_files()):
            if item["is_directory"]:
                print("Directory:", item["name"])
            else:
                print("File:", dir_name + "/" + item["name"])
    
    file_name = "vgg19.pth"
     # Build the remote path
    #source_file_path = dir_name + "/" + file_name
    source_file_path =  file_name

        # Add a prefix to the filename to 
        # distinguish it from the uploaded file
    dest_file_name = "DOWNLOADED_" + file_name

        # Create a ShareFileClient from a connection string
    file_client = ShareFileClient.from_connection_string(
            connection_string, share_name, source_file_path)

    print("Downloading to:", dest_file_name)

        # Open a file for writing bytes on the local system
    with open(dest_file_name, "wb") as data:
            # Download the file from Azure into a stream
        stream = file_client.download_file()
            # Write the stream to the local file
        data.write(stream.readall())
    return 'File downloaded and saved to disk!'

@app.route('/downloadpth_vitb16')
def downloadpth_vit():
    # Verbindung zur Azure File Storage herstellen
    print("TTEST HIIEEER")
    connection_string = "DefaultEndpointsProtocol=https;AccountName=paintinganalyzerstorage;AccountKey=KXI7rZNbuALQYfQUhAlSvTxyHbVp639X28dFrqtYfNLGwAJM87HJ0yVlgjwb5JaKNWl7mpdLI0KR+AStEZd+RA==;EndpointSuffix=core.windows.net"
    #file_service = ShareServiceClient.from_connection_string(conn_str=connection_string)

    # Name des File Shares und des Dateipfades
    share_name = "vgg19"
    #file_path = "vgg19_pretrained_cnn_model_65epoch_best_Acc.pth"
    file_path = "test3.txt"
    dir_name = ""
    # Client für die Datei erstellen und herunterladen
    #share_client = ShareFileClient.from_connection_string(conn_str=connection_string, share_name=share_name, file_path=file_path)
    share_client = ShareClient.from_connection_string(connection_string, share_name)
    print("TEST")
    print(share_client)
    for item in list(share_client.list_directories_and_files()):
            if item["is_directory"]:
                print("Directory:", item["name"])
            else:
                print("File:", dir_name + "/" + item["name"])
    
    file_name = "vitb16.pth"
     # Build the remote path
    #source_file_path = dir_name + "/" + file_name
    source_file_path =  file_name

        # Add a prefix to the filename to 
        # distinguish it from the uploaded file
    dest_file_name = "DOWNLOADED_" + file_name

        # Create a ShareFileClient from a connection string
    file_client = ShareFileClient.from_connection_string(
            connection_string, share_name, source_file_path)

    print("Downloading to:", dest_file_name)

        # Open a file for writing bytes on the local system
    with open(dest_file_name, "wb") as data:
            # Download the file from Azure into a stream
        stream = file_client.download_file()
            # Write the stream to the local file
        data.write(stream.readall())
    return 'File downloaded and saved to disk!'


@app.route('/Get_Pred_Resnet50', methods=['GET'])
def jsonexample():
    print("TEST Hier \n \n")
    print(os.getcwd())
    print("TEST Hier \n \n")
    model = Resnet50_multiTaskNet().to(device) 
    model.load_state_dict(torch.load("flaskml/multitask_resnet_nofreezedlayers_moredataaugmentation_01_0001_0001_weight_null_epoch45_2.pth", map_location=torch.device('cpu')))
    prediction = predict(model, "flaskml/63.jpg")
    print(prediction[3][0][0])
    Dictionary ={'artist': prediction[3][0][0] , 'artist_acc': prediction[0], 
                  'style': prediction[3][1][0] ,'style_acc': prediction[1], 
                  'genre': prediction[3][2][0], 'genre_acc': prediction[2]}
    test = jsonify(Dictionary)
    return test

@app.route('/Get_Pred_Resnet50_2', methods=['POST'])
def resnetfunction():
    content = request.get_json()
    model = Resnet50_multiTaskNet().to(device) 
    model.load_state_dict(torch.load("flaskml/resnet50_pretrained_cnn_model_65epoch_best_Acc.pth", map_location=torch.device('cpu')))
    prediction = predict(model, content['imageurl'])
    print(prediction[3][0][0])
    Dictionary ={'artist': prediction[3][0][0] , 'artist_acc': prediction[0], 
                  'style': prediction[3][1][0] ,'style_acc': prediction[1], 
                  'genre': prediction[3][2][0], 'genre_acc': prediction[2]}
    test = jsonify(Dictionary)
    #return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    return test

@app.route('/Get_Pred_vgg19', methods=['POST'])
def vggfunction():
    content = request.get_json()
    model = VGG_19_multiTaskNet().to(device) 
    model.load_state_dict(torch.load("DOWNLOADED_vgg19.pth", map_location=torch.device('cpu')))
    prediction = predict(model, content['imageurl'])
    print(prediction[3][0][0])
    Dictionary ={'artist': prediction[3][0][0] , 'artist_acc': prediction[0], 
                  'style': prediction[3][1][0] ,'style_acc': prediction[1], 
                  'genre': prediction[3][2][0], 'genre_acc': prediction[2]}
    test = jsonify(Dictionary)
    #return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    return test

@app.route('/Get_Pred_vit_b_16', methods=['POST'])
def vitfunction():
    content = request.get_json()
    model = ViT_multiTaskNet().to(device) 
    model.load_state_dict(torch.load("DOWNLOADED_vitb16.pth", map_location=torch.device('cpu')))
    prediction = predict(model, content['imageurl'], True)
    print(prediction[3][0][0])
    Dictionary ={'artist': prediction[3][0][0] , 'artist_acc': prediction[0], 
                  'style': prediction[3][1][0] ,'style_acc': prediction[1], 
                  'genre': prediction[3][2][0], 'genre_acc': prediction[2]}
    test = jsonify(Dictionary)
    #return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    return test


@app.route('/')
def index():
  return '<h1>I want to Deploy Flask to Azure</h1>'


"""@app.route('/json-example', methods=['GET'])
def jsonexample():
    Dictionary ={'username':'eduCBA' , 'account':'Premium' , 'validity':'2709 days'}
    test = jsonify(Dictionary)
    print(test.response[0])
    my_json = test.response[0].decode('utf8')
    print(my_json)
    print("TEST")
    return test"""

 
