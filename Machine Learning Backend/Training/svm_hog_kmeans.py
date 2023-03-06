import torch
import numpy as np
import cv2
import os
import scipy
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
import skimage.data as skid
import pylab as plt
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import pandas as pd


#Datensatz einlesen
csv_path_train = 'groundtruth_multiloss_train_header.csv' 
csv_path_test = 'groundtruth_multiloss_test_header.csv' 
df = pd.read_csv(csv_path_train)
df1 = pd.read_csv(csv_path_test)
data = [df, df1]
df2 = pd.concat(data)


# takes all images from images Folder
def load_images_from_imagesfolder(folder, extension):
    images = []
    current_path = pathlib.Path(folder).resolve()
    for ext in extension:
        imagesfromfolder = Path(current_path).glob(ext)
        for image in imagesfromfolder:
            images.append(str(image))
    return images

# takes all images from csv
def load_images_from_csv(df):
    images = []
    current_path = str(pathlib.Path('original/images').resolve())
    print(current_path)
    for index, row in df.iterrows():
        image = current_path + '/' + row['filename']
        images.append(str(image))
    return images 

# All images from the folder are being loaded 
folderforimages = "images"
allimages = load_images_from_imagesfolder(folderforimages, ('*.png', '*.jpg'))
allimages = load_images_from_csv(df2)
image_paths_len = len(allimages)
print(type(image_paths_len))
print(f"länge alle images:  {image_paths_len}")


#Die Liste an Descriptoren hat die Dimension : Anzahl der Bilder * 20 * 128
SIFT_Dimension_Per_Image = 128
Keypoints_per_Image = 20

#total_SIFT_features = np.zeros((image_paths_len,Keypoints_per_Image, SIFT_Dimension_Per_Image ))
total_SIFT_features = np.zeros((image_paths_len*Keypoints_per_Image, SIFT_Dimension_Per_Image ))
#total_SIFT_features = np.zeros((20,128))

sift = cv2.xfeatures2d.SIFT_create()

#len(allimages)
def DSIFT_Hist_voacb(input_stepsize, input_n_sample):
#Die Features werden aus den Bildern durch D-SIFT extrahiert
    desc_exception = None
    for i in range(0, len(allimages)):
        if i % 10000 == 0:
            print(i)
        img = cv2.imread(allimages[i])
        try:
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        
            step_size = input_stepsize
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                                for x in range(0, gray.shape[1], step_size)]

            

            #dense_feat = sift.compute(gray, kp)
            [descriptors, keypoints]= sift.compute(gray, kp)
        

            n_sample = input_n_sample
            desc_samples = keypoints[np.random.randint(keypoints.shape[0],  size=n_sample)]
            desc_exception = desc_samples
            total_SIFT_features[(i*n_sample):(i*n_sample)+(n_sample),] = desc_samples[:]
        
        except Exception as e:
            total_SIFT_features[(i*n_sample):(i*n_sample)+(n_sample),] = desc_exception[:]
            print(allimages[i])
            with open("Output.txt", "w") as text_file:
                text_file.write("%s \n" % allimages[i])
        
        
        #print(desc_samples.shape)

step_size = 20
n_sample = 20
vocab_size = 200
#Funktion für D-Sift und Histogramm Erstellung des Vokabulars (Bag of Features)
#Funktion nur benutzen wenn Bag of Features aktualisiert werden soll
DSIFT_Hist_voacb(step_size,n_sample)
kmeans = KMeans(n_clusters=vocab_size, random_state=0)
label = kmeans.fit_predict(total_SIFT_features)
joblib.dump(kmeans, 'my_model_knn2.pkl')
print("test")
exit()

#KNN Model wird geladen und anschließend werden die Clustercenter als Bag of Features verwendet
#kmeans = joblib.load("my_model_knn2.pkl")
#vocab = kmeans.cluster_centers_

print(vocab.shape)



feats = []
sift = cv2.xfeatures2d.SIFT_create()
#Die Features werden nochmal aus den Bildern durch D-SIFT extrahiert und anschließend mit dem vokabular verglichen
image_feats2 = None
for i in range(len(allimages)):
    if i % 1000 == 0:
        print(i)
    try:
        img = cv2.imread(allimages[i])
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        

        #Descriptor mit Dimension 128 x Keypoints , Abhänging von der Größe der Bilder
        sift = cv2.xfeatures2d.SIFT_create()
        step_size = 10
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                            for x in range(0, gray.shape[1], step_size)]

        dense_feat = sift.compute(gray, kp)
        [descriptors, keypoints]= sift.compute(gray, kp)


        #Erschaffung des Histogramms
        # Die euklidische Distanz wird berechnet zwischen den 128 dimensionalen Vektor und dem jeweiligen Cluster Center
        # Aus beispielsweise 7900 keypoints mit jeweils 128-D Vektor , wird die Entfernung von dem Keypoint zu jedem Cluster Center berechnet
        # Jede euklidistische Distanz ist genau ein Wert und somit erhalten wir die Anzahl der cluster als Dimension und die Anzahl der samples
        # Also Beispielsweise 7900 samples x 10 Cluster, die Cluster sind hier die Vokabeln im Sinne des Bag of Words / Features
        dist = scipy.spatial.distance.cdist(keypoints,vocab) 


        #Jedes Sample hat jetzt Sample x Cluster/Vokabel als Dimension also zb 7900 x 10
        #Die Vokabel anzahl wird reduziert auf die mit dem geringsten euklidischen Abstand
        #Daraus ensteht eine 1 Dimensionales Array zb 7900 x 1
        bin_assignment = np.argmin(dist, axis=1)



        image_feats = np.zeros(vocab_size)
        for id_assign in bin_assignment:
            image_feats[id_assign] += 1
    
        image_feats2 = image_feats
        feats.append(image_feats)
    except Exception as e:
            feats.append(image_feats2)
            print(allimages[i])
            with open("Output2.txt", "w") as text_file:
                text_file.write("%s \n" % allimages[i])


#Normalisierung der Vektoren
feats = np.asarray(feats)
print(feats.shape)
feats_norm_div = np.linalg.norm(feats, axis=1)
for i in range(0, feats.shape[0]):
    feats[i] = feats[i] / feats_norm_div[i]

print(feats)
print(feats.shape)

#https://stackoverflow.com/questions/29799053/how-to-print-result-of-clustering-in-sklearn
#440 labels = Jede Zahl bestimm zu welchen cluster das sample gehört , also einfach einen neuen array basteln mit dimension
# 200 x 128 , index ist der label , also  zb variable[label] = total_sift[label] ,np auf zero initalisieren
# variable mit labellengh x 128, unique labels verwenden , dann haste 200 einzigartige 128 vektoren
#SVM mit center trainieren & mit direkten labels


#Number of Images x Number of Clusters 
#Zb Mona lisa hat 10 , Van Gogh hat 10 usw
#Die Vektoren werden in ein SVM gefüttert