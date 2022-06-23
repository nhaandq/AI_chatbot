import numpy as np
import scipy
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

raw_folder = "D:\SPKT/2021_2022_2\AI\Assembly/data/training_data"

image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)    

train_dataset = image_generator.flow_from_directory(batch_size=10,
                                                 directory=raw_folder,
                                                 shuffle=True,
                                                 target_size=(150, 150), 
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=10,
                                                 directory=raw_folder,
                                                 shuffle=True,
                                                 target_size=(150, 150), 
                                                 subset="validation",
                                                 class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150,150,3) ) )
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same') )
model.add(MaxPooling2D( (2,2) ) )

model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same') )
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same') )
model.add(MaxPooling2D( (2,2) ) )

model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same') )
model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same') )
model.add(MaxPooling2D( (2,2) ) )

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30,activation='softmax'))
model.summary()


opt = SGD(learning_rate = 0.001, momentum = 0.9)
model .compile(optimizer = opt, loss ='categorical_crossentropy', metrics = ['accuracy'])

history=model.fit(train_dataset,batch_size=32,epochs=100,verbose=1,validation_data=validation_dataset)
model.save('foodmodel5.h5')

print('Done')