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

for line in lines[1:]:
    source_path = line[0]
    filename = 'data/' + source_path
    image = cv2.imread(filename)
    images.append(image)
    steeting_measurements.append(float(line[3]))

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
from keras.layers import Cropping2D

model = Sequential()

'''
Preprocess the data 
1) Normalize the data by diving each element by 255. Now elements are in range between 0 and 1.
2) Mean center the data by subtracting 0.5 from each element
'''

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

activation = 'relu' 

# define the first set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(20, 5, padding="same"))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the second set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(50, 5, padding="same"))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the first FC => ACTIVATION layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation(activation))
 
# define the second FC layer
model.add(Dense(1))

# No softmax, as this is a regression task and I want my netowrk to directly predict the steering measurement
model.compile(loss = 'mse', optimizer = 'Adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')

exit()
