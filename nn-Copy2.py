A#!/usr/bin/env python
# coding: utf-8

# # **Milestone 1**

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np


# In[ ]:


pip install opencv-python


# In[ ]:


# Libraries for file operations
import zipfile
import os
import cv2


# In[ ]:


# Libraries to display graphs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# In[ ]:


import seaborn as sns


# ### **1. Import the data**

# In[ ]:


# file_paths
car_images_path = r"C:\Users\Acer\Downloads\Car_detection_project\Car_Images.zip"
annotations_file_path = r"C:\Users\Acer\Downloads\Car_detection_project\Annotations.zip"
car_names_file_path = r"C:\Users\Acer\Downloads\Car_detection_project\Car_names_and_make.csv"


# In[ ]:


# Car Names Data

car_names_df = pd.read_csv(car_names_file_path)


# In[ ]:


car_names_df.head(50)


# In[ ]:


# Car Annotations Data

archive = zipfile.ZipFile(annotations_file_path)
archive.extractall()


# In[ ]:


train_annotations_df = pd.read_csv(r"C:\Users\Acer\Downloads\Car_detection_project\Annotations\Train Annotations.csv")
test_annotations_df = pd.read_csv(r"C:\Users\Acer\Downloads\Car_detection_project\Annotations\Test Annotation.csv")


# In[ ]:


train_annotations_df.shape, train_annotations_df.head()


# In[ ]:


test_annotations_df.shape, test_annotations_df.head()


# In[ ]:


# Car Images Data

car_images_archive = zipfile.ZipFile(car_images_path)
car_images_archive.extractall()


# ### **2. Map training and testing images to its classes.**

# In[ ]:


import os
import cv2

# Path to the training images directory
train_images_dataset = r'C:\Users\Acer\Downloads\Car_detection_project\Car_Images\Car Images'

image = []
image_name = []
car_class = []

# Use os.walk() to traverse the directory tree
for root, dirs, files in os.walk(train_images_dataset):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):  # Filter for image files
            car_name = os.path.basename(root)  # Get the car class from directory name
            image_path = os.path.join(root, file)  # Full path to the image
            image_array = cv2.imread(image_path)  # Read the image
            if image_array is not None:  # Check if the image has been successfully loaded
                image.append(image_array)  # Add the image array to the list
                image_name.append(file)  # Add the image file name to the list
                car_class.append(car_name)  # Add the car class to the list
            else:
                print(f"Failed to load image: {image_path}")  # Notify if an image fails to load

# Display the results (optional, can be replaced with any processing/display function)
print("Loaded", len(image), "images.")
if image:
    print("Example image names:", image_name[:5])  # Print first 5 image names
    print("Example classes:", car_class[:5])  # Print classes of the first 5 images


# In[ ]:


# Checking length of each list before creating the training dataframe
print(len(car_class), len(image_name), len(image))


# In[ ]:


train_image_df = pd.DataFrame()
train_image_df['Image_Name'] = image_name
train_image_df['Image_Class'] = car_class
train_image_df['Actual_Image'] = image


# In[ ]:


train_image_df.head()


# In[ ]:


# Number of recors for each classes
num_classes_data = pd.DataFrame(train_image_df['Image_Class'].value_counts())
num_classes_data


# **EDA Train Data**

# In[ ]:


# Distribution of Number of classes
sns.distplot(a=num_classes_data['Image_Class']);


# In[ ]:


# Box plot of Number of classes
sns.boxplot(num_classes_data['Image_Class']);


# In[ ]:


# Finding car names data
num_classes_data.reset_index(inplace=True)


# In[ ]:


# Deriving New Features For Car Company, Car Model, Car Manufacturing Year
company = []
car_model = []
mfg_year = []

for index, row in num_classes_data.iterrows():
  split_row = row['index'].split()
  company.append(split_row[0])
  mfg_year.append(split_row[-1])
  car_model.append(' '.join(split_row[1:-1]))

num_classes_data['Company'] = company
num_classes_data['Car_Model'] = car_model
num_classes_data['Mfg_Year'] = mfg_year


# In[ ]:


# Graph of Car Company vs Count of Car models

car_company_data = pd.DataFrame(num_classes_data['Company'].value_counts())
car_company_data.reset_index(inplace=True)
plt.figure(figsize=(20, 7))
ax = sns.barplot(data=car_company_data, x='index', y='Company')
plt.xticks(rotation=90)
ax.set(xlabel='Car Company', ylabel='Count')
plt.tight_layout()
plt.show()


# **Analysis**
# 
# 1.   Chevrolet has highest numbers of car
# 2.   Maybach, McLaren, Porshe and 16 others company has lowest number of cars
# 
# 

# In[ ]:


# Analysis

# Car models manufactured by "Chevrolet"
num_classes_data[num_classes_data['Company'] == "Chevrolet"][['Car_Model', 'Mfg_Year']]


# In[ ]:


# Graph of Manufacturing Year vs Count of Car models

plt.figure(figsize=(7, 5))
ax = sns.countplot(x=num_classes_data["Mfg_Year"], order=np.unique(num_classes_data["Mfg_Year"]))
plt.xticks(rotation=90)
ax.set(xlabel='Manufacturing Year', ylabel='Count')
plt.tight_layout()
plt.show()


# **Analysis**
# 
# 1.   Oldest car was manufactured in year 1991
# 2.   Earliest car was manufactured in year 2012
# 
# 

# In[ ]:


# Analysis

# 1. Car Manufactured in year "1991"
num_classes_data[num_classes_data['Mfg_Year'] == "1991"]


# In[ ]:


import os
import cv2

# Path to the test images directory
test_images_dataset = r'C:\Users\Acer\Downloads\Car_detection_project\Car_Images\Car Images\Test Images'

test_image = []
test_image_name = []
test_car_class = []

# Use os.walk() to traverse the directory tree
for root, dirs, files in os.walk(test_images_dataset):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):  # Filter for image files
            car_name = os.path.basename(root)  # Get the car class from directory name
            image_path = os.path.join(root, file)  # Full path to the image
            image_array = cv2.imread(image_path)  # Read the image
            if image_array is not None:  # Check if the image has been successfully loaded
                test_image.append(image_array)  # Add the image array to the list
                test_image_name.append(file)  # Add the image file name to the list
                test_car_class.append(car_name)  # Add the car class to the list
            else:
                print(f"Failed to load image: {image_path}")  # Notify if an image fails to load

# Display the results (optional)
print("Loaded", len(test_image), "images.")
if test_image:
    print("Example image names:", test_image_name[:5])  # Print first 5 image names
    print("Example classes:", test_car_class[:5])  # Print classes of the first 5 images


# In[ ]:


print(len(car_class), len(image_name), len(image))  #


# In[ ]:


test_image_df = pd.DataFrame()
test_image_df['Image_Name'] = image_name
test_image_df['Image_Class'] = car_class
test_image_df['Actual_Image'] = image


# In[ ]:


test_image_df.head()


# ### **3. Map training and testing images to its annotations.**

# In[ ]:


# Mapping train images to it's annotations on Image_Name

train_image_df


# In[ ]:


train_annotations_df


# In[ ]:


train_data = train_image_df.merge(train_annotations_df, how="inner", left_on="Image_Name", right_on="Image Name")
train_data


# In[ ]:


# Mapping test images to it's annotations on Image_Name

test_data = test_image_df.merge(test_annotations_df, how="inner", left_on="Image_Name", right_on="Image Name")
test_data


# ### **4. Display images with bounding box.**

# In[ ]:


# This method will display n random images with bounding box

def print_random_images(data, num_images=5):
  upper_range = len(data)
  random_num =  np.random.randint(1, high=upper_range, size=num_images)
  N = len(random_num)

  plt.figure(figsize=(N*5, 5))
  for itr in range(N):
      plt.subplot(1, N, itr+1)
      plt.imshow(data['Actual_Image'][random_num[itr]])  # greens, reds, blues, rgb
      # Add the patch to the Axes
      x = data['Bounding Box coordinates'][random_num[itr]]
      y = data['Unnamed: 2'][random_num[itr]]
      w = data['Unnamed: 3'][random_num[itr]] - x
      h = data['Unnamed: 4'][random_num[itr]] - y
      plt.gca().add_patch(Rectangle((x, y), width=w, height=h, linewidth=2, edgecolor='y', facecolor='none'))
      plt.title("{}".format(data['Image_Class'][random_num[itr]]))
      plt.axis('off')
  plt.show()

print_random_images(train_data, num_images=5)


# ### **5. Design, train and test basic CNN models to classify the car.**

# **Create X & Y from the DataFrame**

# Checking if each image has 3 channels i.e. RGB

# In[ ]:


for index, row in train_data.iterrows():
  if row['Actual_Image'].shape[2] != 3:
    print(index)


# In[ ]:


for index, row in test_data.iterrows():
  if row['Actual_Image'].shape[2] != 3:
    print(index)


# In[ ]:


# resizing image
def resize_image_and_coordinates(row):
  image_array = row['Actual_Image']
  image_array_shape = image_array.shape

  #resizing image
  image_new_size = 128
  resized_image = cv2.resize(image_array, (image_new_size, image_new_size))
  #normalizing image
  resized_image = resized_image.astype('float32')
  resized_image = resized_image / 255.

  resized_image = np.array(resized_image)

  x1_old = row['Bounding Box coordinates']
  y1_old = row['Unnamed: 2']
  x2_old = row['Unnamed: 3']
  y2_old = row['Unnamed: 4']

  x_compression = image_new_size / image_array_shape[1]
  y_compression = image_new_size / image_array_shape[0]

  x1_new = int(x1_old * x_compression)
  y1_new = int(y1_old * y_compression)
  x2_new = int(x2_old * x_compression)
  y2_new = int(y2_old * y_compression)

  resized_bounding_box_cord = [x1_new, y1_new, x2_new, y2_new]

  return resized_image, resized_bounding_box_cord


# **Training data**

# In[ ]:


# iterating over Training data to reshape them
X_train_res = []
X_train_res_bound_box = []
for index, row in train_data.iterrows():
  X_new, X_new_bound_box = resize_image_and_coordinates(row)
  X_train_res.append(X_new)
  X_train_res_bound_box.append(X_new_bound_box)

X_train = np.array(X_train_res)
X_train_bound_box = np.array(X_train_res_bound_box)


# In[ ]:


# Number of classes
train_data['Image class'].nunique()


# In[ ]:


np.unique(train_data['Image class'])


# In[ ]:


y_train = pd.get_dummies(train_data['Image class']).values


# In[ ]:


y_train.shape


# In[ ]:


X_train.shape, y_train.shape


# **Testing data**

# In[ ]:


# iterating over Tresting data to reshape them
X_test_res = []
X_test_res_bound_box = []
for index, row in test_data.iterrows():
  X_new, X_new_bound_box = resize_image_and_coordinates(row)
  X_test_res.append(X_new)
  X_test_res_bound_box.append(X_new_bound_box)

X_test = np.array(X_test_res)
X_test_bound_box = np.array(X_test_res_bound_box)


# In[ ]:


# Number of classes
test_data['Image class'].nunique()


# In[ ]:


np.unique(test_data['Image class'])


# In[ ]:


y_test = pd.get_dummies(test_data['Image class']).values


# In[ ]:


X_test.shape, y_test.shape


# In[ ]:


# this method will display Reshaped  images with bounding box

def print_random_images(data_image_array, data_bounding_box, num_images=5):
  upper_range = len(data_image_array)
  random_num =  np.random.randint(1, high=upper_range, size=num_images)
  N = len(random_num)

  plt.figure(figsize=(N*5, 5))
  for itr in range(N):
      plt.subplot(1, N, itr+1)
      plt.imshow(data_image_array[random_num[itr]])  # greens, reds, blues, rgb
      # Add the patch to the Axes
      x = data_bounding_box[random_num[itr]][0]
      y = data_bounding_box[random_num[itr]][1]
      w = data_bounding_box[random_num[itr]][2] - x
      h = data_bounding_box[random_num[itr]][3] - y
      plt.gca().add_patch(Rectangle((x, y), width=w, height=h, linewidth=2, edgecolor='y', facecolor='none'))
      plt.axis('off')
  plt.show()

print_random_images(X_test, X_test_bound_box, num_images=5)


# In[ ]:


# Deleting redundant variables to free RAM
del archive


# **CNN Model**

# In[ ]:


pip install --upgrade keras


# In[ ]:


from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from keras.layers import Flatten
import tensorflow as tf
from keras.layers import Convolution2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers.legacy import Adam


# In[ ]:


from keras.utils import image_dataset_from_directory


# Using Keras Library to Load and Preprocess data

# In[ ]:


# initializing variables related to image data

batch_size = 32  # batch size
img_height = 160
img_width = 160
train_data_dir = r"C:\Users\Acer\Downloads\Car_detection_project\Car Images\Train Images"
test_data_dir = r"C:\Users\Acer\Downloads\Car_detection_project\Car Images\Test Images"


# In[ ]:


import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123,  # Ensuring reproducibility
    shuffle=True  # Shuffling for training data
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123,  # Same seed to ensure consistent validation split if applicable
    shuffle=False  # Usually, we do not shuffle the test data
)


# In[ ]:


class_names = train_ds.class_names
steps_per_epoch = len(train_ds) // batch_size
validation_steps = len(test_ds) // batch_size


# In[ ]:


# reading test data from directory
val_ds = tf.keras.utils.image_dataset_from_directory(
  test_data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


# Data Augmentation Layer
data_augmentation = tf.keras.Sequential(
  [
    RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.2),
    RandomZoom(0.1),
  ]
)


# In[ ]:


num_classes = 196

model = Sequential([
  data_augmentation,  # adding augmentation layer
  Rescaling(1./255),  # adding rescaling layer
  # 1 Conv Layer
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  # 2 Conv Layer
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Dropout(0.1),
  # 3 Conv Layer
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Dropout(0.2),
  # 4 Conv Layer
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Dropout(0.2),


  Flatten(),

  # 1 Fully Connected Layer
  Dense(128, activation='relu'),
  Dropout(0.1),
  # 1 Fully Connected Layer
  Dense(64, activation='relu'),
  # The final output layer with 196 neuron to predict the categorical classifcation
  Dense(num_classes, name="outputs", activation = 'softmax')
])


# In[ ]:


model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb, early_stopping_cb]
)


# In[ ]:


# Visualizing model performance on Training and Validation data

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # **Milestone 2**

# ### **1. Fine tune the trained basic CNN models to classify the car.**

# In[ ]:


# Basic CNN Model to improve performance

num_classes = 196

model = Sequential([
  # Data Augmentation Layer
  RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
  RandomRotation(0.3),
  RandomZoom(0.2),
  Rescaling(1./255),  # adding rescaling layer
  # 1 Conv Layer
  Conv2D(16, 3, activation='relu'),
  AveragePooling2D(),
  BatchNormalization(),
  # 2 Conv Layer
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  BatchNormalization(),
  Dropout(0.1),
  # 3 Conv Layer
  Conv2D(64, 3, activation='relu'),
  # AveragePooling2D(),
  BatchNormalization(),
  # Dropout(0.2),
  # 4 Conv Layer
  # Conv2D(64, 3, activation='relu'),
  # AveragePooling2D(),
  # BatchNormalization(),
  # Dropout(0.2),


  Flatten(),

  # 1 Fully Connected Layer
  Dense(128, activation='relu'),
  BatchNormalization(),
  Dropout(0.1),
  # 2 Fully Connected Layer
  Dense(64, activation='relu'),
  BatchNormalization(),
  # 3 Fully Connected Layer
  Dense(64, activation='relu'),
  # BatchNormalization(),
  # The final output layer with 196 neuron to predict the categorical classifcation
  Dense(num_classes, name="outputs", activation = 'softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

model.summary()


# In[ ]:


# This callback will stop the training when there is no improvement in the loss for three consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

epochs = 10
history = model.fit(
 train_ds,
   validation_data=val_ds,
   epochs=epochs,
   steps_per_epoch=steps_per_epoch,
   validation_steps=validation_steps,
   callbacks=[checkpoint_cb, early_stopping_cb]
)


# Get 23% training and approximately 6% validation accuracy after adding few dense and Batch normalization layers over basic CNN model

# **Transfer Learning**

# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# In[ ]:


# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# model preprocessing layer
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Global AveragePooling Layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Prediction layer
prediction_layer = tf.keras.layers.Dense(196)


# In[ ]:


IMG_SHAPE = (160, 160) + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base_model
base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


initial_epochs = 10

history = model.fit(train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb, early_stopping_cb]
)


# In[ ]:


base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

epochs = 10
model.fit(train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb, early_stopping_cb]
)


# In[ ]:


model.summary()


# ### **Saving the model**

# In[ ]:


# saving the model
model.save(r"C:\Users\Acer\Downloads\Car_detection_project\car_classification_model.h5")


# In[ ]:


# Need pickle file to save the class names
import pickle


# In[ ]:


# writing class names to a pickle file
with open(r"C:\Users\Acer\Downloads\Car_detection_project\car_classes.pkl", 'wb') as f:
  pickle.dump(class_names, f)


# In[ ]:


pip install colab


# In[ ]:





# In[ ]:


# Making a prediction by loading the model
from matplotlib.pyplot import imshow
model_pred = tf.keras.models.load_model(r"C:\Users\Acer\Downloads\Car_detection_project\car_classification_model.h5")

img_height_pred = 160
img_width_pred = 160
img_path_pred = r"C:\Users\Acer\Downloads\Car_detection_project\Car Images\Test Images\Nissan Juke Hatchback 2012\06807.jpg"
imshow(cv2.imread(img_path_pred))

img = tf.keras.utils.load_img(img_path_pred, target_size=(img_height_pred, img_height_pred))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model_pred.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# ### **2. Design, train and test YOLO**

# In[ ]:


yolo_cfg_path = "/content/drive/MyDrive/Capestone_AIML/Car_Detection/yolov3.cfg"
yolo_weights_path = "/content/drive/MyDrive/Capestone_AIML/Car_Detection/yolov3.weights"
coco_names_path = "/content/drive/MyDrive/Capestone_AIML/Car_Detection/coco.names"

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


# In[ ]:


# img = cv2.imread('/content/Car Images/Train Images/Audi A5 Coupe 2012/01715.jpg')
img = cv2.imread('/content/Car Images/Train Images/Rolls-Royce Ghost Sedan 2012/03250.jpg')


# In[ ]:


# saving image
image_name = "yolo_img"
image_type = "png"
image_name_type = image_name + "." + image_type
cv2.imwrite(image_name_type, img)


# In[ ]:


# construct a blob from the image
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]


# In[ ]:


net.setInput(blob)
outputs = net.forward(ln)


# In[ ]:


boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

# Load names of classes and get random colors
classes = open(coco_names_path).read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2_imshow(img)


# **Original VS Predicted Bounding Box**

# In[ ]:


# Function for Object detection using YOLO

def object_detection_yolo3(image_file):
    file_path = "/content/drive/MyDrive/Capestone_AIML/Car_Detection/"
    yolo_cfg_path = file_path + "yolov3.cfg"
    yolo_weights_path = file_path + "yolov3.weights"
    coco_names_path = file_path + "coco.names"

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(image_file)

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    # Load names of classes and get random colors
    classes = open(coco_names_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


# In[ ]:


def print_object_image(random_index):
  clr = [0, 0, 255]
  plt.figure(figsize=(10, 10))

  row = test_data.iloc[random_index]

  image_dir = "/content/Car Images/Test Images/"
  image_class = row['Image_Class']
  image_name = row['Image_Name']
  image_path = image_dir + image_class + "/" + image_name
  img_array = cv2.imread(image_path)
  x = row['Bounding Box coordinates']
  y = row['Unnamed: 2']
  w = row['Unnamed: 3'] - x
  h = row['Unnamed: 4'] - y
  cv2.rectangle(img_array, (x, y), (x + w, y + h), clr, 2)
  # cv2_imshow(img_array)
  plt.subplot(1, 2, 1)
  plt.imshow(img_array)
  plt.title("Original Bounding Box")
  plt.axis('off')

  yolo_img = object_detection_yolo3(image_path)
  plt.subplot(1, 2, 2)
  plt.imshow(yolo_img)
  plt.title("Predicted Objects")
  plt.axis('off')


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  5393  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  62  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  1098  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  7999  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  4567  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  2356  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  3821  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  100  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  786  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# In[ ]:


# Calling method to plot original and predicted bounding box
random_num =  1009  # upper limit -> 8000 test data lenght
print_object_image(random_num)


# # **Milestone 3**

# **GUI**

# In[ ]:


from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import pickle
import cv2
import numpy as np
import tensorflow as tf


# In[ ]:


def object_detection_yolo3(image_file):
    file_path = "C:\\Users\\Capestone Project\\Project - Car Detection\\"
    yolo_cfg_path = file_path + "yolov3.cfg"
    yolo_weights_path = file_path + "yolov3.weights"
    coco_names_path = file_path + "coco.names"

#     print(image_file)

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(image_file)

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    # Load names of classes and get random colors
    classes = open(coco_names_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    #resizing image
    resized_image = cv2.resize(img, (450, 240))

    return resized_image

def save_object_detect_image(img, image_path):
    # splitting file path on "/"
    image_path_list = image_path.split("/")

    # extracting image name and updating it
    image_name = image_path_list[-1].split(".")[0]
    image_name_with_type = image_name + "_object_detect.png"

    # creating the file path with updated image name
    image_path_without_name = image_path_list[:-1]
    image_path_without_name.append(image_name_with_type)
    object_detect_image_path = "\\".join(image_path_without_name)

    # saving the image
    path_to_store_image = "yolo_img_pred.png"
    cv2.imwrite(object_detect_image_path, img)

    return object_detect_image_path

def pred_image_class(image_path):
    # loading car class pickle file
    car_classes_file_path = "car_classes.pkl"
    with open(car_classes_file_path, 'rb') as f:
        class_names = pickle.load(f)

    # Loading the model and predicting car class
    # Making a prediction by loading the model
    model_path = 'car_classification_model.h5'
    model_pred = tf.keras.models.load_model(model_path)

    img_height_pred = 160
    img_width_pred = 160
    img_path_pred = image_path

    img = tf.keras.utils.load_img(img_path_pred, target_size=(img_height_pred, img_height_pred))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model_pred.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    prediction_str = 'This image belongs to "{}" with {:.2f} confidence'.format(class_names[np.argmax(score)], 100 * np.max(score))

    return prediction_str


# In[ ]:


root = Tk()
root.title("Object Detection")
root.geometry("500x400")

def add_label(image_path):
    global my_label
    car_class = pred_image_class(image_path)
    my_label = Label(root, text=car_class, fg="blue")
    my_label.pack(pady=10)

def add_processing_label():
    global my_processing_label
    my_processing_label = Label(root, text="Loading...", fg="blue")
    my_processing_label.pack(pady=40)

def add_image():
    # Disabling open image button
    open_button['state'] = DISABLED
    # adding pre processing label
    add_processing_label()

    # Add image
    global my_image
    global my_image_label

    # input the image file to predict
    image_path = filedialog.askopenfilename()

    # Object detection on input image and saving that image file
    object_deteced_image = object_detection_yolo3(image_path)
    object_image_path = save_object_detect_image(object_deteced_image, image_path)

    my_image = PhotoImage(file=object_image_path)

    # destroying preprocessing label
    my_processing_label.destroy()

    # Creating a label to display image
    my_image_label = Label(image=my_image)
    my_image_label.pack()

    # Adding label for class prediction
    add_label(image_path)


def delete_label():
    my_label.destroy()
    my_image_label.destroy()

    open_button['state'] = NORMAL

open_button = Button(root, text="Open Image File", command=add_image)
open_button.pack(pady=10)

clear_button = Button(root, text="Clear Image", command=delete_label)
clear_button.pack(pady=10)

# quit_button = Button(root, text="Exit Program", command=root.quit)
# quit_button.pack()

root.mainloop()


# In[ ]:




