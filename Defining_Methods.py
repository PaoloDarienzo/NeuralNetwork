import os, os.path
from subprocess import getoutput
import tensorflow as tf
from tensorflow import keras as ks

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
import cv2
#Data visualization
import seaborn as sns
from matplotlib import pyplot as plt

import glob
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import time
import math

from collections import Counter

from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax

def get_dataset_crop(db, _batch_size, _dim1, _dim2, drive):

  if db == '9k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1nderD97u_2d1I6TE3ey8wBtEWqLAosk4"})
    downloaded.GetContentFile('data_9400_88.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_9400_88.zip -d DB_Repo/
    getoutput("unzip -q data_9400_88.zip -d DB_Repo/")

  elif db == '14k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1z5J7XE_KJYzZGJd-NHX8PUggED2ZK8HG"})
    downloaded.GetContentFile('data.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data.zip -d DB_Repo/
    getoutput("unzip -q data.zip -d DB_Repo/")

  elif db == '18k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"18ESID3MpwG-SzZPE1EENzsGPh8vl8ti9"})
    downloaded.GetContentFile('data_18800.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_18800.zip -d DB_Repo/
    getoutput("unzip -q data_18800.zip -d DB_Repo/")

  else:
    raise ValueError("Keyword for database not recognized; use '9k', '14k' or '18k'.")
  
  path, dirs, files = next(os.walk("/content/DB_Repo/data"))
  file_count = len(files)
  print(file_count)
  #%cd /content/DB_Repo/data
  os.chdir("/content/DB_Repo/data")

  ##PARAMETERS

  dim1 = _dim1
  dim2 = _dim2

  total_pixels = dim1 * dim2
  MAX_LEN = 64 #fisso

  #Considero il primo 20% della lista di dati come test set
  test_percentage = 20 #%
  #Considero il 20% della lista di dati - esclusi i dati di test - come validation set
  validation_percentage = 20 #%

  #COSTANTI E DICHIARAZIONI

  database_list = list()
  labels_list = list()
  obf_list = list()

  #LETTURA E RESIZE IMMAGINI

  print("START IMAGE INPUT")
  #Aggiungo i valori alle liste leggendo i vari files
  for filename in glob.glob('*.npy'):
    temp_img = np.load(filename)
    temp_img = temp_img.reshape((-1, MAX_LEN)).astype(np.uint16)
    #flattening
    temp_img = temp_img.flatten()
    dimensione = temp_img.size

    #padding fino alla dimensione dim1xdim2
    #o crop fino a dim1xdim2 pixels
    if dimensione < total_pixels:
      temp_img = np.pad(temp_img, (0, total_pixels - dimensione), mode='constant',constant_values=0)
    elif dimensione >= total_pixels:
      temp_img = temp_img[0:total_pixels]
    else:
      raise ValueError("Error in reading images.")

    temp_img = temp_img.reshape((dim1, dim2))
    database_list.append(temp_img)
    #Salvo la label, ossia la classe
    labels_list.append(extract_label(filename))
    #Salvo la lista di offuscatori di ogni file
    obf_list.append(extract_obf(filename))
  print("END IMAGE INPUT")

  #SHUFFLE

  #Ho i valori e le etichette in due liste (+ obf); 
  #le mescolo mantenendo l'ordine tra valore-label
  temp = list(zip(database_list, labels_list, obf_list))
  np.random.shuffle(temp)
  database_list, labels_list, obf_list = zip(*temp)

  #SUDDIVISIONE DATI
  #Suddivido in training set, test set e validation test
  assert len(database_list) == len(labels_list) == len(obf_list)
  #print(len(database_list))

  #Split per creare test set
  index_to_split = math.ceil((len(database_list) * test_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(database_list) - 1)]

  test_list, training_list = [database_list[s:e+1] for s,e in indices]
  labels_test_list, labels_training_list = [labels_list[s:e+1] for s,e in indices]
  obf_test_list, obf_training_list = [obf_list[s:e+1] for s,e in indices]

  #Split per creare validation set
  index_to_split = math.ceil((len(training_list) * validation_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(training_list) - 1)]

  validation_list, training_list = [training_list[s:e+1] for s,e in indices]
  labels_validation_list, labels_training_list = [labels_training_list[s:e+1] for s,e in indices]
  obf_validation_list, obf_training_list = [obf_training_list[s:e+1] for s,e in indices]

  #Trasformo i valori in numpy.ndarray
  train_images = np.array(training_list)
  test_images = np.array(test_list)
  validation_images = np.array(validation_list)

  train_labels = np.array(labels_training_list)
  test_labels = np.array(labels_test_list)
  validation_labels = np.array(labels_validation_list)

  train_obf = np.array(obf_training_list)
  test_obf = np.array(obf_test_list)
  validation_obf = np.array(obf_validation_list)

  #Encoding delle labels;
  #Se nella suddivisione il 100% di una classe è fuori dal train_labels,
  #Vi sarà un errore nell'encoding delle labels negli altri set.
  label_encoder = LabelEncoder()
  label_encoder.fit(train_labels)
  train_labels_encoded = label_encoder.transform(train_labels)
  test_labels_encoded = label_encoder.transform(test_labels)
  validation_labels_encoded = label_encoder.transform(validation_labels)

  #Normalizzazione valori in range 0-1
  train_images = train_images / 65535.0
  test_images = test_images / 65535.0
  validation_images = validation_images / 65535.0

  #Dichiarazione altri parametri
  n_classes = len(list(label_encoder.classes_))

  sets_and_labels = (train_images, train_labels_encoded, test_images, test_labels_encoded, validation_images, validation_labels_encoded)
  numpy_arrays = (train_obf, test_obf, validation_obf)
  return sets_and_labels, numpy_arrays, label_encoder, n_classes

def get_dataset_crop_v2(db, _batch_size, _dim1, _dim2, n_features, drive):

  if db == '9k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1nderD97u_2d1I6TE3ey8wBtEWqLAosk4"})
    downloaded.GetContentFile('data_9400_88.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_9400_88.zip -d DB_Repo/
    getoutput("unzip -q data_9400_88.zip -d DB_Repo/")

  elif db == '14k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1z5J7XE_KJYzZGJd-NHX8PUggED2ZK8HG"})
    downloaded.GetContentFile('data.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data.zip -d DB_Repo/
    getoutput("unzip -q data.zip -d DB_Repo/")

  elif db == '18k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"18ESID3MpwG-SzZPE1EENzsGPh8vl8ti9"})
    downloaded.GetContentFile('data_18800.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_18800.zip -d DB_Repo/
    getoutput("unzip -q data_18800.zip -d DB_Repo/")

  else:
    raise ValueError("Keyword for database not recognized; use '9k', '14k' or '18k'.")
  
  path, dirs, files = next(os.walk("/content/DB_Repo/data"))
  file_count = len(files)
  print(file_count)
  #%cd /content/DB_Repo/data
  os.chdir("/content/DB_Repo/data")

  ##PARAMETERS

  dim1 = _dim1
  dim2 = _dim2

  total_pixels = dim1 * dim2
  MAX_LEN = 64 #fisso

  #Considero il primo 20% della lista di dati come test set
  test_percentage = 20 #%
  #Considero il 20% della lista di dati - esclusi i dati di test - come validation set
  validation_percentage = 20 #%

  #COSTANTI E DICHIARAZIONI

  database_list = list()
  labels_list = list()
  obf_list = list()

  #LETTURA E RESIZE IMMAGINI

  print("START IMAGE INPUT")
  #Aggiungo i valori alle liste leggendo i vari files
  for filename in glob.glob('*.npy'):
    temp_img = np.load(filename)
    temp_img = temp_img.reshape((-1, MAX_LEN)).astype(np.uint16)
    #flattening
    temp_img = temp_img.flatten()
    dimensione = temp_img.size

    #padding fino alla dimensione dim1xdim2
    #o crop fino a dim1xdim2 pixels
    if dimensione < total_pixels:
      temp_img = np.pad(temp_img, (0, total_pixels - dimensione), mode='constant',constant_values=0)
    elif dimensione >= total_pixels:
      temp_img = temp_img[0:total_pixels]
    else:
      raise ValueError("Error in reading images.")

    #temp_img = temp_img.reshape((dim1, dim2))
    temp_img = temp_img.reshape((-1, n_features))
    database_list.append(temp_img)
    #Salvo la label, ossia la classe
    labels_list.append(extract_label(filename))
    #Salvo la lista di offuscatori di ogni file
    obf_list.append(extract_obf(filename))
  print("END IMAGE INPUT")

  #SHUFFLE

  #Ho i valori e le etichette in due liste (+ obf); 
  #le mescolo mantenendo l'ordine tra valore-label
  temp = list(zip(database_list, labels_list, obf_list))
  np.random.shuffle(temp)
  database_list, labels_list, obf_list = zip(*temp)

  #SUDDIVISIONE DATI
  #Suddivido in training set, test set e validation test
  assert len(database_list) == len(labels_list) == len(obf_list)
  #print(len(database_list))

  #Split per creare test set
  index_to_split = math.ceil((len(database_list) * test_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(database_list) - 1)]

  test_list, training_list = [database_list[s:e+1] for s,e in indices]
  labels_test_list, labels_training_list = [labels_list[s:e+1] for s,e in indices]
  obf_test_list, obf_training_list = [obf_list[s:e+1] for s,e in indices]

  #Split per creare validation set
  index_to_split = math.ceil((len(training_list) * validation_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(training_list) - 1)]

  validation_list, training_list = [training_list[s:e+1] for s,e in indices]
  labels_validation_list, labels_training_list = [labels_training_list[s:e+1] for s,e in indices]
  obf_validation_list, obf_training_list = [obf_training_list[s:e+1] for s,e in indices]

  #Trasformo i valori in numpy.ndarray
  train_images = np.array(training_list)
  test_images = np.array(test_list)
  validation_images = np.array(validation_list)

  train_labels = np.array(labels_training_list)
  test_labels = np.array(labels_test_list)
  validation_labels = np.array(labels_validation_list)

  train_obf = np.array(obf_training_list)
  test_obf = np.array(obf_test_list)
  validation_obf = np.array(obf_validation_list)

  #Encoding delle labels;
  #Se nella suddivisione il 100% di una classe è fuori dal train_labels,
  #Vi sarà un errore nell'encoding delle labels negli altri set.
  label_encoder = LabelEncoder()
  label_encoder.fit(train_labels)
  train_labels_encoded = label_encoder.transform(train_labels)
  test_labels_encoded = label_encoder.transform(test_labels)
  validation_labels_encoded = label_encoder.transform(validation_labels)

  #Normalizzazione valori in range 0-1
  train_images = train_images / 65535.0
  test_images = test_images / 65535.0
  validation_images = validation_images / 65535.0

  #Dichiarazione altri parametri
  n_classes = len(list(label_encoder.classes_))

  sets_and_labels = (train_images, train_labels_encoded, test_images, test_labels_encoded, validation_images, validation_labels_encoded)
  numpy_arrays = (train_obf, test_obf, validation_obf)
  return sets_and_labels, numpy_arrays, label_encoder, n_classes

#NB: cv2 inverts rows and cols wrt numpy
def get_dataset_interp(db, _batch_size, _dim1, _dim2, drive):

  if db == '9k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1nderD97u_2d1I6TE3ey8wBtEWqLAosk4"})
    downloaded.GetContentFile('data_9400_88.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_9400_88.zip -d DB_Repo/
    getoutput("unzip -q data_9400_88.zip -d DB_Repo/")

  elif db == '14k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"1z5J7XE_KJYzZGJd-NHX8PUggED2ZK8HG"})
    downloaded.GetContentFile('data.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data.zip -d DB_Repo/
    getoutput("unzip -q data.zip -d DB_Repo/")

  elif db == '18k':
    #%cd /content
    os.chdir("/content")
    downloaded = drive.CreateFile({'id':"18ESID3MpwG-SzZPE1EENzsGPh8vl8ti9"})
    downloaded.GetContentFile('data_18800.zip')
    #!rm -rf DB_Repo
    getoutput("rm -rf DB_Repo")
    #!unzip -q data_18800.zip -d DB_Repo/
    getoutput("unzip -q data_18800.zip -d DB_Repo/")

  else:
    raise ValueError("Keyword for database not recognized; use '9k', '14k' or '18k'.")
  
  path, dirs, files = next(os.walk("/content/DB_Repo/data"))
  file_count = len(files)
  print(file_count)
  #%cd /content/DB_Repo/data
  os.chdir("/content/DB_Repo/data")

  ##PARAMETERS

  dim1 = _dim1
  dim2 = _dim2

  #total_pixels = dim1 * dim2
  MAX_LEN = 64 #fisso

  #Considero il primo 20% della lista di dati come test set
  test_percentage = 20 #%
  #Considero il 20% della lista di dati - esclusi i dati di test - come validation set
  validation_percentage = 20 #%

  #COSTANTI E DICHIARAZIONI

  database_list = list()
  labels_list = list()
  obf_list = list()

  #LETTURA E RESIZE IMMAGINI

  print("START IMAGE INPUT")
  #Aggiungo i valori alle liste leggendo i vari files
  for filename in glob.glob('*.npy'):
    temp_img = np.load(filename)
    temp_img = temp_img.reshape((-1, MAX_LEN)).astype('float32') 
    temp_img = cv2.resize(temp_img, (dim2, dim1), interpolation=cv2.INTER_CUBIC)
    database_list.append(temp_img)
    #Salvo la label, ossia la classe
    labels_list.append(extract_label(filename))
    #Salvo la lista di offuscatori di ogni file
    obf_list.append(extract_obf(filename))
  print("END IMAGE INPUT")

  #SHUFFLE

  #Ho i valori e le etichette in due liste (+ obf); 
  #le mescolo mantenendo l'ordine tra valore-label
  temp = list(zip(database_list, labels_list, obf_list))
  np.random.shuffle(temp)
  database_list, labels_list, obf_list = zip(*temp)

  #SUDDIVISIONE DATI
  #Suddivido in training set, test set e validation test
  assert len(database_list) == len(labels_list) == len(obf_list)
  #print(len(database_list))

  #Split per creare test set
  index_to_split = math.ceil((len(database_list) * test_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(database_list) - 1)]

  test_list, training_list = [database_list[s:e+1] for s,e in indices]
  labels_test_list, labels_training_list = [labels_list[s:e+1] for s,e in indices]
  obf_test_list, obf_training_list = [obf_list[s:e+1] for s,e in indices]

  #Split per creare validation set
  index_to_split = math.ceil((len(training_list) * validation_percentage) / 100)
  indices = [(0, index_to_split - 1), (index_to_split, len(training_list) - 1)]

  validation_list, training_list = [training_list[s:e+1] for s,e in indices]
  labels_validation_list, labels_training_list = [labels_training_list[s:e+1] for s,e in indices]
  obf_validation_list, obf_training_list = [obf_training_list[s:e+1] for s,e in indices]

  #Trasformo i valori in numpy.ndarray
  train_images = np.array(training_list)
  test_images = np.array(test_list)
  validation_images = np.array(validation_list)

  train_labels = np.array(labels_training_list)
  test_labels = np.array(labels_test_list)
  validation_labels = np.array(labels_validation_list)

  train_obf = np.array(obf_training_list)
  test_obf = np.array(obf_test_list)
  validation_obf = np.array(obf_validation_list)

  #Encoding delle labels;
  #Se nella suddivisione il 100% di una classe è fuori dal train_labels,
  #Vi sarà un errore nell'encoding delle labels negli altri set.
  label_encoder = LabelEncoder()
  label_encoder.fit(train_labels)
  train_labels_encoded = label_encoder.transform(train_labels)
  test_labels_encoded = label_encoder.transform(test_labels)
  validation_labels_encoded = label_encoder.transform(validation_labels)

  #Normalizzazione valori in range 0-1
  train_images = train_images / 65535.0
  test_images = test_images / 65535.0
  validation_images = validation_images / 65535.0

  #Dichiarazione altri parametri
  n_classes = len(list(label_encoder.classes_))

  sets_and_labels = (train_images, train_labels_encoded, test_images, test_labels_encoded, validation_images, validation_labels_encoded)
  numpy_arrays = (train_obf, test_obf, validation_obf)
  return sets_and_labels, numpy_arrays, label_encoder, n_classes

#Extract the class from the file name, if the class is the string before che -
def extract_label(from_string):
  position = from_string.index('-') # gets position of the - in the filename
  substring = from_string[0:position]
  return substring

def extract_obf(from_string):
  start_pos = from_string.index('-')
  end_pos = from_string.index('.')
  substring = from_string[(start_pos + 1):end_pos]
  return substring

def mapping_labels_encoded(label_encoder):
  for index in range(len(list(label_encoder.classes_))):
    print(index, end = "-> ")
    print(list(label_encoder.inverse_transform([index]))) 

class TimeHistory(ks.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_model_acc(hist):
  fig = plt.figure()
  #Plot training & validation accuracy values
  plt.plot(hist.history['acc'])
  plt.plot(hist.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  return fig
  #plt.show()

def plot_model_loss(hist):
  fig = plt.figure()
  #Plot training & validation accuracy values
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  return fig
  #plt.show()

def plot_conf_matrix(modelLSTM, validation_images, validation_labels_encoded, label_encoder):
  #Necessito di un array con tutte le labels
  validation_predictions = modelLSTM.predict_classes(validation_images)

  conf_matr = confusion_matrix(y_true = validation_labels_encoded, y_pred = validation_predictions)
  #print(conf_matr)

  con_mat_norm = np.around(conf_matr.astype('float') / conf_matr.sum(axis=1)[:, np.newaxis], decimals=2)

  con_mat_df = pd.DataFrame(con_mat_norm,
                          index = list(label_encoder.classes_), 
                          columns = list(label_encoder.classes_))

  figure = plt.figure(figsize=(len(list(label_encoder.classes_)), len(list(label_encoder.classes_))), dpi=50)
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure 
  #plt.show()

def stampa_grafo_orizzontale(dict_errori, grandezza_x, grandezza_y, x_label, y_label, colore):
  fig = plt.figure(figsize=(grandezza_x, grandezza_y))

  plt.bar(list(dict_errori.keys()), list(dict_errori.values()), color=colore)

  plt.axhline(np.asarray(list(dict_errori.values())).mean(), color="red") # Horizontal line adding the threshold
  #plt.axhline(np.asarray(list(dict_errori.values())).std(), color="grey") # Horizontal line adding the threshold
  plt.axhline(np.asarray(list(dict_errori.values())).max(), color="black") # Horizontal line adding the threshold

  plt.xlabel(x_label) # x label
  plt.ylabel(y_label) # y label
  return fig
  #plt.show()

def stampa_grafo_verticale(dict_errori, grandezza_x, grandezza_y, x_label, y_label, colore):
  fig = plt.figure(figsize=(grandezza_x, grandezza_y))

  plt.barh(list(dict_errori.keys()), list(dict_errori.values()), color=colore)

  plt.axvline(np.asarray(list(dict_errori.values())).mean(), color="red") # Horizontal line adding the threshold
  #plt.axvline(np.asarray(list(dict_errori.values())).std(), color="grey") # Horizontal line adding the threshold
  plt.axvline(np.asarray(list(dict_errori.values())).max(), color="black") # Horizontal line adding the threshold

  plt.xlabel(x_label) # x label
  plt.ylabel(y_label) # y label
  return fig
  #plt.show()

def computing_incorrects_stats(modelLSTM, validation_images, validation_labels_encoded, validation_obf, label_encoder):
  incorrects = np.nonzero(modelLSTM.predict_classes(validation_images) != validation_labels_encoded)

  temp_incorrects = list()
  for elem in incorrects[0]:
    temp_incorrects.append(elem)
  incorrects = temp_incorrects

  wrong_labels_str = list()
  wrong_obf = list()
  for elem in incorrects:
    string_to_append = str(label_encoder.inverse_transform([validation_labels_encoded[elem]]))
    wrong_labels_str.append(string_to_append)
    wrong_obf.append(validation_obf[elem])
  
  assert len(wrong_labels_str) == len(wrong_obf)

  set_obfs = list()
  for elem in wrong_obf:
    temp_list = elem.split('-')  
    temp_list.sort()
    separator = '-'
    temp_list = separator.join(temp_list)
    set_obfs.append(temp_list)

  single_obf = list()
  for elem in wrong_obf:
    temp_list = elem.split('-')
    for sub_elem in temp_list:
      single_obf.append(sub_elem)

  single_obfs_total = list()

  for elem in validation_obf:
    temp_list = elem.split('-')

    for sub_elem in temp_list:
      single_obfs_total.append(sub_elem)


  count_labels_err = Counter(wrong_labels_str)
  count_obf_err = Counter(wrong_obf)
  count_set_obfs = Counter(set_obfs)

  count_single_obfs_total = Counter(single_obfs_total)
  count_single_obf = Counter(single_obf)

  single_obf_percentage = dict()

  for key, value in count_single_obfs_total.items():
    error_val = count_single_obf.get(key)
    if type(error_val)==None.__class__:
      percentuale = 0
    else:
      percentuale = (100 * error_val) / value
    single_obf_percentage.update({key : percentuale})

  return count_labels_err, count_obf_err, count_set_obfs, single_obf_percentage

def modelVanilla(num_units1, num_units2, _batch_size, n_classes, _patience, sub_db, n_epochs):

  (train_images, train_labels_encoded, test_images, test_labels_encoded) = sub_db

  modelLSTM = ks.Sequential()

  #Batch size should be (at most) the same number of hidden cells
  #no activation selection

  modelLSTM.add(Flatten())
  modelLSTM.add(tf.keras.layers.Dense(num_units1))
  modelLSTM.add(tf.keras.layers.Dense(num_units2))
  modelLSTM.add(Dense(n_classes, activation='softmax'))

  modelLSTM.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  #Definizione callback
  es = ks.callbacks.EarlyStopping(monitor='val_loss', patience=_patience,
                                  mode='auto', restore_best_weights=True, verbose=0)
  time_callback = TimeHistory()

  #Validation_data è usato al termine di ogni epoch;
  hist = modelLSTM.fit(train_images, train_labels_encoded, 
                      batch_size = _batch_size,
                      validation_data=(test_images, test_labels_encoded), 
                      epochs=n_epochs, shuffle='true',
                      callbacks=[time_callback, es], verbose=0)

  return modelLSTM, hist, time_callback

def modelLSTM(num_units1, num_units2, time_steps, n_features, _batch_size, n_classes, _patience, sub_db, n_epochs):

  (train_images, train_labels_encoded, test_images, test_labels_encoded) = sub_db

  modelLSTM = ks.Sequential()

  #Batch size should be (at most) the same number of hidden cells
  #no activation selection
  modelLSTM.add(Bidirectional(CuDNNLSTM(num_units1, unit_forget_bias='true', return_sequences='true'),
                              input_shape=(time_steps, n_features)))
  modelLSTM.add(Bidirectional(CuDNNLSTM(num_units2, unit_forget_bias='true')))
  modelLSTM.add(Dense(n_classes, activation='softmax'))

  modelLSTM.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  #Definizione callback
  es = ks.callbacks.EarlyStopping(monitor='val_loss', patience=_patience,
                                  mode='auto', restore_best_weights=True, verbose=0)
  time_callback = TimeHistory()

  #Validation_data è usato al termine di ogni epoch;
  hist = modelLSTM.fit(train_images, train_labels_encoded, 
                      batch_size = _batch_size,
                      validation_data=(test_images, test_labels_encoded), 
                      epochs=n_epochs, shuffle='true',
                      callbacks=[time_callback, es], verbose=0)

  return modelLSTM, hist, time_callback

def reshape_for_ConvLSTM2D(sub_db, time_steps, n_features, _channels):

  (train_images, test_images, validation_images) = sub_db

  channels = _channels
  #Reshape degli array di immagini
  n_img, _, _ = train_images.shape
  n_img2, _, _ = test_images.shape
  n_img3, _, _ = validation_images.shape

  train_images = train_images.reshape(n_img, -1, time_steps, n_features, channels)
  test_images = test_images.reshape(n_img2, -1, time_steps, n_features, channels)
  validation_images = validation_images.reshape(n_img3, -1, time_steps, n_features, channels)

  _, n_ts_blocks, _, _, _ = train_images.shape
  
  return train_images, test_images, validation_images, n_ts_blocks

def modelConvLSTM2D(num_units1, time_steps, n_features, n_ts_blocks, _batch_size, channels, n_classes, _patience, sub_db, n_epochs):

  (train_images, train_labels_encoded, test_images, test_labels_encoded) = sub_db

  modelLSTM = ks.Sequential()

  #Batch size should be (at most) the same number of hidden cells
  #no activation selection
  modelLSTM.add(Bidirectional(ConvLSTM2D(num_units1, (3, 3),
                              padding='same', unit_forget_bias='true', activation='relu'), 
                              input_shape=(n_ts_blocks, time_steps, n_features, channels)))
  modelLSTM.add(Flatten())
  modelLSTM.add(Dense(n_classes, activation='softmax'))

  modelLSTM.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  #Definizione callback
  es = ks.callbacks.EarlyStopping(monitor='val_loss', patience=_patience,
                                  mode='auto', restore_best_weights=True, verbose=0)
  time_callback = TimeHistory()

  #Validation_data è usato al termine di ogni epoch;
  hist = modelLSTM.fit(train_images, train_labels_encoded, 
                      batch_size = _batch_size,
                      validation_data=(test_images, test_labels_encoded), 
                      epochs=n_epochs, shuffle='true',
                      callbacks=[time_callback, es], verbose=0)
  
  return modelLSTM, hist, time_callback