import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import keras.applications.inception_v3 as iv3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

iv3.WEIGHTS_PATH = 'http://papaleguas.icomp.ufam.edu.br/~marco/downloads/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
iv3.WEIGHTS_PATH_NO_TOP = 'http://papaleguas.icomp.ufam.edu.br/~marco/downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = iv3.InceptionV3(weights='imagenet')

def reset_graph(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
	
dir_data = '/home/nmarc/tensorflow/mcm/c3d-keras-tensorflow/c3d-keras/Folds_Dataset_Final'

# dimensions of our images.
#img_width, img_height = 50, 50
img_width, img_height = 299, 299


train_data_dir = 'Folds_Dataset_Final/libras/train' 
validation_data_dir = 'Folds_Dataset_Final/libras/validation' 

# create the base pre-trained model
base_model = iv3.InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(40, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
	
	
	
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

datagen = ImageDataGenerator(
        rotation_range=40,       # image rotation (degrees)
        width_shift_range=0.2,   # width-% range for image translation 
        height_shift_range=0.2,  # height-% range for image translation 
        shear_range=0.2,         # cf https://en.wikipedia.org/wiki/Shear_mapping
        zoom_range=0.2,          # zoom
        horizontal_flip=True,    # mirroring effect
        fill_mode='nearest')     # how to fill gaps resulting from other transformations

img = load_img(sinais[random.randint(0, len(sinais)-1)])
x = img_to_array(img)


batch_size = 10

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255)#,
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

start_time = time.time()

print "start history model"
history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=1000, # 50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
duration = time.time() - start_time

model.save_weights('./log/libras_rev3.h5')

