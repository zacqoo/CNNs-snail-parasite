"""
Created on Wed Dec 20 10:18:30 2017
@author: Zac Yung-Chun Liu
"""
# import libraries and packages
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
# import cv2
# ------------------------------------------------------------------------------  
# define parameters
# dimensions of our images.
# input_shape
in_s = 64  
img_width, img_height = in_s, in_s  
  
top_model_weights_path = 'bottleneck_fc_model.h5'  
train_data_dir = 'dataset/training_set'  
validation_data_dir = 'dataset/test_set'  
# number of epochs to train top model  
epochs = 20  
# batch size used by flow_from_directory and predict_generator  
batch_size = 32
# ------------------------------------------------------------------------------  
# create the VGG16 model - 
# without the final fully-connected layers (by specifying include_top=False) 
# - and load the ImageNet weights
model = applications.VGG16(include_top=False, weights='imagenet') 
# ------------------------------------------------------------------------------  
# create the data generator for training images, 
# and run them on the VGG16 model to save the bottleneck features for training
train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   #rotation_range=30, 
                                   #width_shift_range=0.2, 
                                   #height_shift_range=0.2, 
                                   horizontal_flip = False)
   
generator = train_datagen.flow_from_directory(  
    train_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    class_mode=None,  
    shuffle=False)  
  
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  
  
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
  
bottleneck_features_train = model.predict_generator(  
    generator, predict_size_train)  
  
np.save('bottleneck_features_train.npy', bottleneck_features_train)
# ------------------------------------------------------------------------------  
# do the same for the validation data
datagen = ImageDataGenerator(rescale= 1./255) 

generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
  
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
  
bottleneck_features_validation = model.predict_generator(  
    generator, predict_size_validation)  
  
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
# ------------------------------------------------------------------------------  
# to train the top model, need the class labels for each of the training/validation samples 
# also need to convert the labels to categorical vectors
train_datagen_top = ImageDataGenerator(rescale= 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       #rotation_range=30, 
                                       #width_shift_range=0.2, 
                                       #height_shift_range=0.2, 
                                       horizontal_flip = False)

generator_top = train_datagen_top.flow_from_directory(  
        train_data_dir,  
        target_size=(img_width, img_height),  
        batch_size=batch_size,  
        class_mode='categorical',  
        shuffle=False)  
   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  
  
# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
  
# get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  
  
# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes) 
# ------------------------------------------------------------------------------  
# do the same for validation features as well
datagen_top = ImageDataGenerator(rescale=1./255)

generator_top = datagen_top.flow_from_directory(  
         validation_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames)  
  
validation_data = np.load('bottleneck_features_validation.npy')  
  
validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)  
# ------------------------------------------------------------------------------  
# create and train a small fully-connected network - 
# the top model - using the bottleneck features as input, with our classes as the classifier output
model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.6))  
model.add(Dense(num_classes, activation='softmax'))  
  
model.compile(optimizer='rmsprop',  
             loss='categorical_crossentropy', metrics=['accuracy'])  
  
history = model.fit(train_data, train_labels,  
         epochs=epochs,  
         batch_size=batch_size,  
         validation_data=(validation_data, validation_labels))  
  
model.save_weights(top_model_weights_path)  
  
(eval_loss, eval_accuracy) = model.evaluate(  
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss))
# ------------------------------------------------------------------------------  
# graph the training history
plt.figure(1)  
   
 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
# ------------------------------------------------------------------------------     
model.load_weights(top_model_weights_path)
pred = model.predict(validation_data,batch_size=64,verbose=1)  
# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(validation_data)

from numpy import argmax
# invert encoding
val_labels = []
for i in range(0,len(class_predicted)):
    val_labels.append(argmax(validation_labels[i]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_labels, class_predicted)
print(cm)
accuracy = (cm[0,0] +cm[1,1]+cm[2,2]+cm[3,3]) / sum(sum(cm))
print(accuracy)

from collections import Counter
Counter(val_labels)
# ------------------------------------------------------------------------------  
# Plot confusion matrix
class_names = [0,1,2,3]
from plot_confusion_matrix import plot_confusion_matrix

# Plot non-normalized confusion matrix
import matplotlib.pyplot as plt
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('cm.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('cm_norm.png')
#plt.show()
# ------------------------------------------------------------------------------  
# ready to train the model
# call these two functions in sequence
#save_bottlebeck_features()  
#train_top_model()     
# ------------------------------------------------------------------------------  
# Make prediction for test set
# apply to the whole test folder
import numpy as np
#from keras.preprocessing import image
import os

test_files = [(root + '/' + filename) for root, directories, filenames in 
              os.walk('dataset/test_set') for filename in filenames]
#test_labels = [2 if "Lymnaea" in f 1 else if "Bulinid" in f else 0 for f in test_files]
test_labels = []
for f in test_files:
    if "Biomphalaria" in f:
        test_labels.append(0)
    if "Bulinus_globtrunc" in f:
        test_labels.append(1)
    if "Lymnaea" in f:
        test_labels.append(2)
    if "Melanoides" in f:
        test_labels.append(3)

def predict_image(image_path, model, target_size=(in_s,in_s)):
    test_image = load_img(image_path, target_size = target_size)
    test_image = img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0) # axis = 0 adds new dimension at the first position, as expected for the batch dimension
    # run the image through the same pipeline
    # build the VGG16 network  
    model = applications.VGG16(include_top=False, weights='imagenet')      
    # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_prediction = model.predict(test_image)  
    # build top model  
    model = Sequential()  
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
    model.add(Dense(256, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(num_classes, activation='sigmoid'))
    model.load_weights(top_model_weights_path)
    # use the bottleneck prediction on the top model to get the final classification  
    class_predicted = model.predict_classes(bottleneck_prediction)
    #prediction = model.predict(bottleneck_prediction)
    return class_predicted

test_predictions = []
for test_file in test_files :
    prediction = predict_image(test_file, model, target_size=(in_s,in_s))
    test_predictions.append(prediction)
    
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1] + cm[2,2] +cm[3,3]) / sum(sum(cm))
print(accuracy)

# Plot confusion matrix
class_names = [0,1,2,3]
from plot_confusion_matrix import plot_confusion_matrix

# Plot non-normalized confusion matrix
import matplotlib.pyplot as plt
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# ------------------------------------------------------------------------------  
# Make prediction for signle image
image_path = 'dataset/eva.2.png'     
#orig = cv2.imread(image_path)  
  
#print("[INFO] loading and preprocessing image...")  
image = load_img(image_path, target_size=(in_s, in_s))  
image = img_to_array(image)   
# important! otherwise the predictions will be '0'  
image = image/255   
image = np.expand_dims(image, axis=0) 
# ------------------------------------------------------------------------------  
# run the image through the same pipeline
# build the VGG16 network  
model = applications.VGG16(include_top=False, weights='imagenet')  
  
# get the bottleneck prediction from the pre-trained VGG16 model  
bottleneck_prediction = model.predict(image)  
  
# build top model  
model = Sequential()  
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='sigmoid'))  
  
model.load_weights(top_model_weights_path)  
  
# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction)
#prediction = model.predict(bottleneck_prediction)  
# ------------------------------------------------------------------------------  
# decode the prediction and show the result
inID = class_predicted[0]  
   
class_dictionary = generator_top.class_indices  
  
inv_map = {v: k for k, v in class_dictionary.items()}  
  
label = inv_map[inID]  
  
# get the prediction label  
print("Image ID: {}, Label: {}".format(inID, label))  
  
# display the predictions with the image  
#cv2.putText(orig, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)  
  
#cv2.imshow("Classification", orig)  
#cv2.waitKey(0)        