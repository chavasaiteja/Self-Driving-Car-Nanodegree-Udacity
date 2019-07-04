import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("******* The Top 5 lines in CSV file are *******")
for line in lines[0:5]:
    print(line)
    print()

images = []
steeting_measurements = []

for line in lines[1:]:
    source_path = line[0]
    filename = 'data/' + source_path
    image = cv2.imread(filename)
    images.append(image)
    steeting_measurements.append(float(line[3]))

# Converting to Numpy arrays as that's the format keras requires
X_train = np.array(images)
y_train = np.array(steeting_measurements)
                                 
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
                                
model = Sequential()

'''
Preprocess the data 
1) Normalize the data by diving each element by 255. Now elements are in range between 0 and 1.
2) Mean center the data by subtracting 0.5 from each element
'''
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))

# No softmax, as this is a regression task and I want my netowrk to directly predict the steering measurement
model.compile(loss = 'mse', optimizer = 'Adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

model.save('model.h5')
          
                                 
                                 