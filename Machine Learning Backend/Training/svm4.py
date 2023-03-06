from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import ast
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
from datetime import timedelta
since = time.time()
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

# Laden der CSV-Datei in einen pandas DataFrame
csv_path_train = "multitask100k_svm2.csv"
df = pd.read_csv(csv_path_train, converters={'X': from_np_array})


# Extrahieren von Spalte "X" und "Y_artist"
X_train = df['X']
Y_artist = df['Y_style']


# Umwandeln der Arrays in Listen und Reshaping
X_train = np.array([x.tolist()[0] for x in X_train])
X_train = X_train.reshape(-1, 1)

# Aufteilen der Daten in Trainings- und Testdatensatz mit 70% für das Training
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_artist, test_size=0.3)

# SVM-Modell initialisieren und trainieren
clf = svm.SVC(kernel='linear', C=1, verbose=True)
clf.fit(X_train, y_train)

# Vorhersage auf dem Testdatensatz
y_pred = clf.predict(X_test)

time_elapsed = time.time() - since
# Auswertung der Vorhersage mit der accuracy_score Funktion
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Erstellen des Balkendiagramms
fig, ax = plt.subplots()
ax.bar(['Accuracy'], [accuracy])

# Setzen der Achsenbeschriftungen und des Titels
ax.set_ylabel('Accuracy')
ax.set_title('SVM Accuracy')

# Speichern des Diagramms als PNG-Bild
plt.savefig('svm_accuracy_style.png')