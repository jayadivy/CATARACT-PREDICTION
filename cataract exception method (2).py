#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


pip install --upgrade pip


# In[3]:


pip install tensorflow


# In[4]:


pip install tensorflow


# In[5]:


pip install keras


# In[6]:


import keras


# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[8]:


import cv2
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[9]:


data = pd.read_csv("full_df.csv")
data.head(20)


# In[10]:


def has_condn(term,text):
    if term in text:
        return 1
    else:
        return 0


# In[11]:


#aching labels based whether cataract is present on which eye (left/right)

import pandas as pd
import matplotlib.pyplot as plt

gender_counts = data['Patient Sex'].value_counts()
colors=['skyblue','pink']
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,colors=colors)

plt.title('Distribution of Patient Sex ')

plt.axis('equal')
plt.show()


# In[12]:


filtered_data = data[(data['Left-Diagnostic Keywords'] == 'normal fundus') & (data['Right-Diagnostic Keywords'] == 'normal fundus')]

gender_counts = filtered_data['Patient Sex'].value_counts()
colors=['skyblue','pink']
gender_counts.plot(kind='bar', rot=0,color=colors)

plt.xlabel('Patient Sex')
plt.ylabel('Count')
plt.title('Distribution of Male and Female Patients with Normal Fundus')

plt.show()


# In[13]:


senior_citizens = data[(data['Left-Diagnostic Keywords'] == 'normal fundus') & (data['Right-Diagnostic Keywords'] == 'normal fundus') &(data['Patient Age'] >= 65)]
colors=['skyblue','pink']
gender_counts.plot(kind='bar', rot=0,color=colors)

plt.xlabel('Patient Sex')
plt.ylabel('Count')
plt.title('Distribution of Male and Female senior citizens with normal fundus ')
plt.show()


# In[14]:


#data[data.O==1].head(20)  #drusen or epiretinal membrane

def process_dataset(data):
    #create 2 more columns labelling them whether right or left cataract
    data["left_cataract"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("cataract",x))
    data["right_cataract"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("cataract",x))
  
    data["LD"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))
    data["RD"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))

    data["LG"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma",x))
    data["RG"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma",x))
    
    data["LH"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))
    data["RH"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))

    data["LM"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("myopia",x))
    data["RM"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("myopia",x))
    
    data["LA"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))
    data["RA"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))
    
    data["LO"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))
    data["RO"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))
    
    #store the right/left cataract images ids in a array
    left_cataract_images = data.loc[(data.C ==1) & (data.left_cataract == 1)]["Left-Fundus"].values
    right_cataract_images = data.loc[(data.C == 1) & (data.right_cataract == 1)]["Right-Fundus"].values
  
    #store the left/right normal image ids in a array
    left_normal = data.loc[(data.C == 0) & (data["Left-Diagnostic Keywords"] == "normal fundus")]['Left-Fundus'].sample(350,random_state=42).values
    right_normal = data.loc[(data.C == 0) & (data["Right-Diagnostic Keywords"] == "normal fundus")]['Right-Fundus'].sample(350,random_state=42).values
    
    #store the left/right diabetes image ids
    left_diab = data.loc[(data.C == 0) & (data.LD == 1)]["Left-Fundus"].values
    right_diab = data.loc[(data.C == 0) & (data.RD == 1)]["Right-Fundus"].values 

    #store the left/right glaucoma image ids
    left_glaucoma = data.loc[(data.C == 0) & (data.LG == 1)]["Left-Fundus"].values
    right_glaucoma = data.loc[(data.C == 0) & (data.RG == 1)]["Right-Fundus"].values 
    
    #store the left/right diabetes image ids
    left_hyper = data.loc[(data.C == 0) & (data.LH == 1)]["Left-Fundus"].values
    right_hyper = data.loc[(data.C == 0) & (data.RH == 1)]["Right-Fundus"].values 
    
    #store the left/right diabetes image ids
    left_myopia = data.loc[(data.C == 0) & (data.LM == 1)]["Left-Fundus"].values
    right_myopia = data.loc[(data.C == 0) & (data.RM == 1)]["Right-Fundus"].values 
    
    #store the left/right diabetes image ids
    left_age = data.loc[(data.C == 0) & (data.LA == 1)]["Left-Fundus"].values
    right_age = data.loc[(data.C == 0) & (data.RA == 1)]["Right-Fundus"].values 
    
    #store the left/right diabetes image ids
    left_other = data.loc[(data.C == 0) & (data.LO == 1)]["Left-Fundus"].values
    right_other = data.loc[(data.C == 0) & (data.RO == 1)]["Right-Fundus"].values 
    
    normalones = np.concatenate((left_normal,right_normal),axis = 0);
    cataractones = np.concatenate((left_cataract_images,right_cataract_images),axis = 0);
    diabones = np.concatenate((left_diab,right_diab),axis = 0);
    glaucoma = np.concatenate((left_glaucoma,right_glaucoma),axis = 0);
    hyper = np.concatenate((left_hyper,right_hyper),axis = 0);
    myopia = np.concatenate((left_myopia,right_myopia),axis = 0);
    age = np.concatenate((left_age,right_age),axis=0);
    other = np.concatenate((left_other,right_other),axis = 0);
    
    return normalones,cataractones,diabones,glaucoma,hyper,myopia,age,other;


# In[15]:


normal , cataract , diab, glaucoma , hyper , myopia , age, other = process_dataset(data);

print("Dataset stats::")
print("Normal ::" , len(normal))
print("Cataract ::" , len(cataract))
print("Diabetes ::" , len(diab))
print("Glaucoma ::" , len(glaucoma))
print("Hypertension ::" , len(hyper))
print("Myopia ::" , len(myopia))
print("Age Issues ::" , len(age))
print("Other ::" , len(other))


# In[16]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array
dataset_dir = "C:/Users/MCA/Desktop/preprocess"
image_size=224
labels = []
dataset = []
def dataset_generator(imagecategory , label):
    for img in tqdm(imagecategory):
        imgpath = os.path.join(dataset_dir,img);
        
        #now we try to read the image and resize it accordingly
        try:
            image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
        except:
            continue;
        dataset.append([np.array(image),np.array(label)]);
    random.shuffle(dataset);
    
    return dataset;


# In[17]:


dataset = dataset_generator(normal,0)
dataset = dataset_generator(cataract,1)
dataset = dataset_generator(diab,2)
dataset = dataset_generator(glaucoma,3)
dataset = dataset_generator(hyper,4)
dataset = dataset_generator(myopia,5)
dataset = dataset_generator(age,6)
dataset = dataset_generator(other,7)

len(dataset)


# In[18]:


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    
    if category== 0:
        label = "Normal"
    elif category == 1 :
        label = "Cataract"
    elif category == 2:
        label = "Diabetes"
    elif category == 3:
        label = "Glaucoma"
    elif category == 4:
        label = "Hypertension"
    elif category == 5:
        label = "Myopia"
    elif category == 6:
        label = "Age Issues"
    else:
        label = "Other"
           
    plt.subplot(2,6,i+1)
    plt.imshow(image)
    plt.xlabel(label)
plt.tight_layout()    


# In[19]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Assuming train_x and train_y are defined as predictors and target respectively
train_x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3);
train_y = np.array([i[1] for i in dataset])
# Splitting the dataset into training, validation, and testing sets
x_train, x_temp, y_train, y_temp = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Convert target labels to categorical format
y_train_cat = to_categorical(y_train, num_classes=8)
y_val_cat = to_categorical(y_val, num_classes=8)
y_test_cat = to_categorical(y_test, num_classes=8)

# Now, x_train, y_train_cat, x_val, y_val_cat, x_test, y_test_cat are prepared for training and evaluation


# In[20]:


datagen = ImageDataGenerator(
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    brightness_range=[0.4, 0.9],  # randomly adjust brightness of images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False,  # do not flip images vertically
    data_format='channels_last',  # data format
    fill_mode='constant'  # fill mode for newly created pixels
)

# Fit the data augmentation generator on the training data
datagen.fit(x_train)


# In[21]:


pip install torch torchvision torchaudio


# In[22]:


conda install pytorch torchvision torchaudio cpuonly -c pytorch


# In[23]:


conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


# In[24]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())


# In[25]:


from __future__ import absolute_import, division, print_function, unicode_literals

# Import libraries
import json
import os
import pickle
import random
import time

# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from skimage import io, transform

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[26]:


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# In[27]:


get_ipython().system('nvidia-smi')


# In[28]:


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

print(device_name)


# In[29]:


def save_checkpoint(state, is_best, filename='/kaggle/working/bt_resnet50_ckpt_v2.pth.tar'):
    torch.save(state, filename)


# In[30]:


from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers

# Assuming you have defined image_size and loaded x_train, y_train_cat, x_test, y_test_cat

xception = Xception(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

# Set all parameters as trainable
for layer in xception.layers:
    layer.trainable = True

# Get the number of input features for the fc layer
n_inputs = xception.output_shape[1]

# Redefine fc layer for the classification problem
xception_top = models.Sequential([
    layers.Flatten(),
    layers.Dense(2048, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(2048, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(8, activation="softmax")
])

# Concatenate Xception base model with the new top layers
model = models.Sequential([
    xception,
    xception_top
])

# Print model summary
model.summary()


# In[31]:


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Compile the model
model.compile(optimizer=SGD(learning_rate=3e-4, momentum=0.9),
              loss="categorical_crossentropy",
              metrics=['accuracy'])


# In[32]:


from sklearn.utils.class_weight import compute_class_weight

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False  # randomly flip images vertically
)

# Fit the data augmentation generator on the training data
datagen.fit(x_train)

# Compute class weights manually
class_counts = np.bincount(y_train)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * class_counts)

# Convert class weights to dictionary format
class_weight = dict(enumerate(class_weights))


# In[39]:


from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',   # Monitor training loss
    patience=10,       # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights to the best epoch
)

# Train the model with data augmentation and class weights
history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=32),
                    steps_per_epoch=len(x_train) / 32,
                    epochs=200,  # Set a large number of epochs
                    validation_data=(x_val, y_val_cat),
                    callbacks=[early_stopping_callback])


# In[35]:


# Adjust the batch size
batch_size = 16

# Train the model with data augmentation and class weights
history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    epochs=200,
                    validation_data=(x_val, y_val_cat),
                    callbacks=[early_stopping_callback])


# In[ ]:


















# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False  # randomly flip images vertically
)

# Assume x_train and y_train are numpy arrays or lists of image paths and labels respectively
# Example usage with a generator
train_generator = datagen.flow_from_directory(
    'C:/Users/admin/Downloads/archive/ODIR-5K/ODIR-5K/Training Images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # or 'categorical' for multi-class classification
)

# Compute class weights manually if using numpy arrays for labels
y_train = np.array([0, 1, 0, 1, 1, 0, 1, 0])  # Example labels, replace with your own labels
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(class_weights))

# Now you can use the train_generator with your model
model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 32,
    epochs=10,
    class_weight=class_weight
)


# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False  # randomly flip images vertically
)

# Fit the data augmentation generator on the training data
datagen.fit(x_train)

# Compute class weights manually
class_counts = np.bincount(y_train)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * class_counts)

# Convert class weights to dictionary format
class_weight = dict(enumerate(class_weights))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False  # randomly flip images vertically
)

# Directory containing the training images, structured by class
train_dir = 'C:/Users/admin/Downloads/archive/ODIR-5K/ODIR-5K/Training Images'

# Verify directory exists
if not os.path.isdir(train_dir):
    raise ValueError(f"Directory {train_dir} does not exist")

# Example usage with a generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # for multi-class classification
)

# Example labels (replace with your own labels)
# Note: this example is for demonstration. You should use actual labels from your dataset.
y_train = np.array([0, 1, 0, 1, 1, 0, 1, 0])  
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(class_weights))

# Now you can use the train_generator with your model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    class_weight=class_weight
)


# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # randomly apply shear transformation
    zoom_range=0.2,  # randomly zoom in/out on images
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False  # randomly flip images vertically
)

# Fit the data augmentation generator on the training data
datagen.fit(x_train)

# Compute class weights manually
class_counts = np.bincount(y_train)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * class_counts)

# Convert class weights to dictionary format
class_weight = dict(enumerate(class_weights))


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',   # Monitor training loss
    patience=10,       # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights to the best epoch
)

# Train the model with data augmentation and class weights
history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=32),
                    steps_per_epoch=len(x_train) / 32,
                    epochs=200,  # Set a large number of epochs
                    validation_data=(x_val, y_val_cat),
                    callbacks=[early_stopping_callback])


# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test_cat)
print("Test Accuracy:", accuracy)

val_loss, val_accuracy = model.evaluate(x_val, y_val_cat)
print("Validation Accuracy:", val_accuracy)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# y_pred = np.array((model.predict(x_test) > 0.5).astype("int32"))

y_pred = []
for i in model.predict(x_test):
    y_pred.append(np.argmax(np.array(i)).astype("int32"))

print(y_pred)


# In[ ]:


plt.figure(figsize=(12,7))
for i in range(20):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    
    if category== 0:
        label = "Normal"
    elif category == 1 :
        label = "Cataract"
    elif category == 2:
        label = "Diabetes"
    elif category == 3:
        label = "Glaucoma"
    elif category == 4:
        label = "Hypertension"
    elif category == 5:
        label = "Myopia"
    elif category == 6:
        label = "Age Issues"
    else:
        label = "Other"
        
    if pred_category== 0:
        pred_label = "Normal"
    elif pred_category == 1 :
        pred_label = "Cataract"
    elif pred_category == 2:
        pred_label = "Diabetes"
    elif pred_category == 3:
        pred_label = "Glaucoma"
    elif pred_category == 4:
        pred_label = "Hypertension"
    elif pred_category == 5:
        pred_label = "Myopia"
    elif pred_category == 6:
        pred_label = "Age Issues"
    else:
        pred_label = "Other"
        
    plt.subplot(4,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 


# In[ ]:


model.save("my_trained_model.h5")


# In[ ]:


from tensorflow.keras.models import load_model
model = load_model('my_trained_model.h5')


# In[ ]:


new_image_path = '/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/1020_left.jpg' 
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224)) 
new_image = new_image / 255.0  

# Perform inference
predictions = model.predict(np.expand_dims(new_image, axis=0))

# Get the class with the highest probability
predicted_class = np.argmax(predictions)

# Now 'predicted_class' contains the predicted class label for the new image
print(f"Actual: {label}")
print(f"Predicted class: {pred_label}")


# In[ ]:


get_ipython().system('zip -r file.zip /kaggle/working')


# In[ ]:


from IPython.display import FileLink
FileLink(r'file.zip')


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_cat)
class_names = ['Normal', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Age Issues', 'Other']

# Print the results
print("Loss:", loss)
print("Accuracy:", accuracy)

# Get predictions on the test set
y_pred = model.predict(x_test)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true class labels
y_true_classes = np.argmax(y_test_cat, axis=1)

# Generate a classification report
report = classification_report(y_true_classes, y_pred_classes)

# Print the classification report
print(report)

# Generate a confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()

# Print the total number of samples for each class
class_counts = {}
for i in range(len(class_names)):
    class_counts[class_names[i]] = sum(y_true_classes == i)

print("Total samples per class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")


# In[ ]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_cat)
class_names = ['Normal', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Age Issues', 'Other']

# Print the results
print("Loss:", loss)
print("Accuracy:", accuracy)

# Get predictions on the test set
y_pred = model.predict(x_test)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true class labels
y_true_classes = np.argmax(y_test_cat, axis=1)

# Generate a classification report
report = classification_report(y_true_classes, y_pred_classes)

# Print the classification report
print(report)

# Generate a confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_cat)
class_names = ['Normal', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Age Issues', 'Other']

# Print the results
print("Loss:", loss)
print("Accuracy:", accuracy)

# Get predictions on the test set
y_pred = model.predict(x_test)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true class labels
y_true_classes = np.argmax(y_test_cat, axis=1)

# Generate a classification report
report = classification_report(y_true_classes, y_pred_classes)

# Print the classification report
print(report)


# In[ ]:





# In[ ]:





