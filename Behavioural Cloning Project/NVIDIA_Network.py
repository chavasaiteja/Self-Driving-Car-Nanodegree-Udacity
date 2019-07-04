import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("\n******* The Top 5 lines in CSV file are *******")
for line in lines[0:5]:
    print(line)

images = []
steeting_measurements = []

# create adjusted steering measurements for the side camera images
correction = 0.2 

for line in lines[1:]:
    center_image_path = line[0]
    left_image_path = line[1]
    right_image_path = line[2]
    
    center_image_path = 'data/' + center_image_path
    left_image_path = 'data/' + left_image_path[1:]
    right_image_path = 'data/' + right_image_path[1:]
    
    steering_center = float(line[3])
    center_image = cv2.imread(center_image_path)
    images.append(center_image)
    steeting_measurements.append(steering_center)
    
    steering_left = steering_center + correction
    left_image = cv2.imread(left_image_path)
    images.append(left_image)
    steeting_measurements.append(steering_left)

    steering_right = steering_center - correction
    right_image = cv2.imread(right_image_path)
    images.append(right_image)
    steeting_measurements.append(steering_right)
 
print(np.array(images).shape)

# Data Augmentation
augmented_images, augmented_steering_measurements = [], [] 

for image, steering_measurement in zip(images, steeting_measurements):
    augmented_images.append(image)
    augmented_steering_measurements.append(steering_measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_steering_measurements.append(steering_measurement*-1.0)
    
# Converting to Numpy arrays as that's the format keras requires
X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Cropping2D, Dropout

model = Sequential()

'''
Preprocess the data 
1) Normalize the data by diving each element by 255. Now elements are in range between 0 and 1.
2) Mean center the data by subtracting 0.5 from each element
'''

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

activation = 'relu' 

# NVIDIA Model
model.add(Conv2D(24, 5))
model.add(Activation(activation))
model.add(Conv2D(36, 5))
model.add(Activation(activation))
model.add(Conv2D(48, 5))
model.add(Activation(activation))
model.add(Conv2D(64, 3))
model.add(Activation(activation))
model.add(Conv2D(64, 3))
model.add(Activation(activation))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation= activation))
model.add(Dense(50, activation= activation))
model.add(Dense(10, activation= activation))
model.add(Dense(1))
    
# No softmax, as this is a regression task and I want my netowrk to directly predict the steering measurement
model.compile(loss = 'mse', optimizer = 'Adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 2)

model.save('model.h5')

exit()