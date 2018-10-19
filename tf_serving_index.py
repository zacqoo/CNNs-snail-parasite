"""
@author: Zac Yung-Chun Liu
"""
# import dependencies
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

# import libraries and packages
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math 

# set tensorflow session
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(0)  # all new operations will be in test mode from now on

#Set variables and randomize things a little
model_version = "1" #Change this to export different model versions, i.e. 2, ..., 7
epoch = 100 ## the higher this number is the more accurate the prediction will be 5000 is a good number to set it at just takes a while to train

seed = 7
np.random.seed(seed)

# ------------------------------------------------------------------------------  
# define parameters
# dimensions of our images.
# input_shape
in_s = 64  
img_width, img_height = in_s, in_s  
  
top_model_weights_path = 'bottleneck_fc_model_93.h5'  
train_data_dir = 'dataset/training_set'  
validation_data_dir = 'dataset/test_set'  
# number of epochs to train top model  
#epochs = 40 
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
#train_datagen = ImageDataGenerator(rescale= 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   #rotation_range=30, 
#                                   #width_shift_range=0.2, 
#                                   #height_shift_range=0.2, 
#                                   horizontal_flip = False)
   
#generator = train_datagen.flow_from_directory(  
#    train_data_dir,  
#    target_size=(img_width, img_height),  
#    batch_size=batch_size,  
#    class_mode=None,  
#    shuffle=False)  
  
#nb_train_samples = len(generator.filenames)  
#num_classes = len(generator.class_indices)  
  
#predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
  
#bottleneck_features_train = model.predict_generator(  
#    generator, predict_size_train)  
  
#np.save('bottleneck_features_train.npy', bottleneck_features_train)
# ------------------------------------------------------------------------------  
# do the same for the validation data
#datagen = ImageDataGenerator(rescale= 1./255) 

#generator = datagen.flow_from_directory(  
#     validation_data_dir,  
#     target_size=(img_width, img_height),  
#     batch_size=batch_size,  
#     class_mode=None,  
#     shuffle=False)  
   
#nb_validation_samples = len(generator.filenames)  
  
#predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
  
#bottleneck_features_validation = model.predict_generator(  
#    generator, predict_size_validation)  
  
#np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
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
model.add(Dropout(0.55))  
model.add(Dense(num_classes, activation='softmax'))

#This is the part that had no documentation or example for keras on how to save the model using keras in the proper format for tensorflow serving
serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x': tf.FixedLenFeature(shape=[4], dtype=tf.float32),}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
#end

x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
y = model(x)

# compile  
model.compile(optimizer='rmsprop',  
             loss='categorical_crossentropy', metrics=['accuracy'])  
  
model.fit(train_data, train_labels,  
         epochs=epochs,  
         batch_size=batch_size,  
         validation_data=(validation_data, validation_labels))

(eval_loss, eval_accuracy) = model.evaluate(  
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss))

# Classification and Prediction signatures for TF serving
values, indices = tf.nn.top_k(y, len(labels))
table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(labels))
prediction_classes = table.lookup(tf.to_int64(indices))

classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

classification_signature = tf.saved_model.signature_def_utils.classification_signature_def(
    examples=serialized_tf_example,
    classes=prediction_classes,
    scores=values
)

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction":y})  

# Check if these signatures are valid
valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
valid_classification_signature = tf.saved_model.signature_def_utils.is_valid_signature(classification_signature)

if(valid_prediction_signature == False):
    raise ValueError("Error: Prediction signature not valid!")

if(valid_classification_signature == False):
    raise ValueError("Error: Classification signature not valid!")

# Save the model
builder = saved_model_builder.SavedModelBuilder('./'+model_version)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  
#model.save_weights(top_model_weights_path)  

builder = saved_model_builder.SavedModelBuilder('./'+model_version)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict-iris':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph
builder.save()
# ------------------------------------------------------------------------------  