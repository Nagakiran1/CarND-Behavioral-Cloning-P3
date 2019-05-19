import pickle
import numpy as np
import os
import pandas as pd
import cv2
np.random.seed(531)
from keras.layers import Input,Dense,Dropout,Conv1D,Conv2D,Activation,LSTM,MaxPooling2D,Flatten, Cropping2D, Lambda
from keras.models import Sequential, Model,model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf as ktf
import keras
import h5py
from IPython.core.debugger import set_trace
import tensorflow as tf
from IPython.core.debugger import set_trace
from sklearn.model_selection import train_test_split

test_ratio = 0.2
valid_ratio = 0.2
batch_size = 64





def PreprocessImages(Driving_log,FolderPath):
    CenterImages = np.concatenate([cv2.imread(FolderPath+file)[np.newaxis] for file in Driving_log[Driving_log.columns[0]].tolist()])
    LeftImages = np.concatenate([cv2.imread(FolderPath+file)[np.newaxis] for file in Driving_log[Driving_log.columns[1]].tolist()])
    RightImages = np.concatenate([cv2.imread(FolderPath+file)[np.newaxis] for file in Driving_log[Driving_log.columns[2]].tolist()])

    CenterImages = CenterImages[:,70:-25,:,:]
    LeftImages = LeftImages[:,70:-25,:,:]
    RightImages = RightImages[:,70:-25,:,:]

    CenterImages = CenterImages/255 - 0.5
    LeftImages = LeftImages/255 - 0.5
    RightImages = RightImages/255 - 0.5

    #X_data = np.concatenate((LeftImages,CenterImages,RightImages),axis=1)
    X_data = CenterImages
    y_data = Driving_log[Driving_log.columns[3]].values[:,None]

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_ratio, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=42)
    return x_train, x_test, x_valid, y_train, y_test, y_valid

#x_train, x_test, x_valid, y_train, y_test, y_valid = PreprocessImages(Driving_log,FolderPath)

#with open('ProcessedData.data','wb') as file:
#    pickle.dump([x_valid,y_valid],file)
#    #pickle.dump([x_train,x_test,x_valid],file)
#    #pickle.dump([y_train,y_test,y_valid],file)
#    file.close()








# def Convolution_block(X,Filters):
#     X_input=X
#     f1,f2,f3,f4=Filters
#     X = Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu')(X_input)
#     X = Dropout(0.5)(X)
    
#     X = Conv2D(filters=f2, kernel_size=8, padding='same', activation='sigmoid')(X)
#     X = Dropout(0.5)(X)
    
#     X = Conv2D(filters=f3, kernel_size=1, padding='same', activation='sigmoid')(X)
#     X = Dropout(0.5)(X)
    
#     X_shortcut=Dense(f4,activation='sigmoid')(X_input)
    
#     X=keras.layers.Add()([X,X_shortcut])
#     X=Activation('relu')(X)
#     return X

# def Identity_block(X,Filters):
#     X_input=X
#     f1,f2,f3=Filters
#     X = Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu')(X_input)
#     X = Dropout(0.5)(X)
    
#     X = Conv2D(filters=f2, kernel_size=8, padding='same', activation='sigmoid')(X)
#     X = Dropout(0.5)(X)
    
#     X = Dense(f3,activation='softmax')(X)
#     X = Dropout(0.5)(X)
    
#     X = keras.layers.Add()([X,X_input])
#     X = Activation('relu')(X)
#     return X

def ConvolutionalModel(input_shape):
    model = Sequential()

    # Convolutional
    model.add(Cropping2D(cropping=((45, 5), (0, 0)), input_shape=input_shape))
    Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    # Fully connected
    model.add(Flatten())
    # model.add(Dropout(0.35))
    model.add(Dense(units=1164))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    # model.add(Lambda(lambda x: np.sign(x) * np.max(abs(x), 90)))

    # Model training

    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['mae'])
    return model

# def Pattern_Recognion_Model(X_train,y_train):
#     X_input = Input(X_train.shape)
    
#     X = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(X_input)

#     X = Dense(512,activation='sigmoid')(X)
    
#     #X = Convolution_block(X,[512,512,248,248])
#     #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
#     #X = Identity_block(X,[248,248,248])
#     #X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
    
#     X = Convolution_block(X,[248,248,124,124])
#     X = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(X)
#     X = Identity_block(X,[124,124,124])
#     X = keras.layers.Flatten()(X)
#     X = Dense(1024, activation='relu')(X)
#     X = Dense(512, activation='relu')(X)
#     X = Dense(64, activation='relu')(X)
#     X=Dense(y_train.shape[1], activation='relu')(X)
    
#     model=Model(inputs=X_input, outputs=X, name='Convolve')
#     model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['mse'])
#     return model

# def Pattern_Recognion_Model2(X_train, y_train, Filters=None):
#     X_input = Input(X_train.shape)
#     f1,f2,f3,f4=Filters
    
#     X = Conv2D(filters=248, kernel_size=3, padding='same', activation='relu')(X_input)
#     X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    
#     X = Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu')(X)
#     X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    
#     X = Dropout(0.5)(X)
    
#     #X = Conv2D(filters=f2, kernel_size=8, padding='same', activation='sigmoid')(X)
#     #X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
#     X = Dropout(0.5)(X)
    
#     #X = Conv2D(filters=f3, kernel_size=1, padding='same', activation='sigmoid')(X)
#     ##X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
#     #X = Dropout(0.5)(X)
#     X = Flatten()(X)
#     X_shortcut=Dense(f4,activation='sigmoid')(X_input)


#     X = Dense(512,activation='sigmoid')(X)
#     X = Dense(64,activation='relu')(X)
#     X=Dense(y_train.shape[1], activation='relu')(X)
    
#     model=Model(inputs=X_input, outputs=X, name='Convolve')
#     model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['mse'])
#     return model


def Pattern_Recognion_Model3(X_train, y_train, Filters=None):
    f1,f2,f3,f4=Filters
    
    #X_input = Input(X_train.shape)
    #X = Sequential()(X_input)
    #X = Cropping2D(cropping=((45, 5), (0, 0)), input_shape=X_train.shape)(Lambda(lambda image: ktf.image.resize_images(image, (80, 200))))
    #X = Cropping2D(cropping=((45, 5), (0, 0)))(X)
    
    X_input = Input(shape=X_train.shape, name='img_in')
    X = X_input
    X = Cropping2D(cropping=((60,25), (0,0)))(X)
    
    X = Lambda(lambda x: (x / 255.0) - 0.5)(X)
    X = Conv2D(filters=16, kernel_size=5, padding='valid', activation='relu')(X)
    X = Conv2D(filters=32, kernel_size=5, padding='valid', activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    X = Conv2D(filters=32, kernel_size=5, padding='valid', activation='relu')(X)
    X = Conv2D(filters=16, kernel_size=5, strides=(2,2), padding='valid', activation='relu')(X)
    X = Conv2D(filters=6, kernel_size=3, padding='valid', activation='relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    X = Conv2D(filters=6, kernel_size=3, strides=(2,2), padding='valid', activation='relu')(X)
    X = Conv2D(filters=6, kernel_size=3, padding='valid', activation='relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    #X = Conv2D(filters=16, kernel_size=3, strides= (2,2), padding='valid', activation='relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(X)
    #X = Conv2D(filters=6, kernel_size=5, strides= (2,2), padding='valid', activation='relu')(X)
    #X = MaxPooling2D(pool_size=(3, 3), strides=(3,3), padding='valid', data_format=None)(X)
    
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(f4,activation='relu')(X)
    X = Dense(248,activation='relu')(X)
    X = Dense(64,activation='relu')(X)
    X = Dense(y_train.shape[1], activation='relu')(X)
    
    model=Model(inputs=X_input, outputs=X, name='Convolve')
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['mse'])
    return model




 


train_datagen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   zca_epsilon=1e-6,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0,
                                   zoom_range=0.1,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   rescale=None,
                                   preprocessing_function=None,
                                   data_format="channels_last")



#x_train, x_test, x_valid, y_train, y_test, y_valid = PreprocessImages(Driving_log,FolderPath)

with open('/opt/ProcessedData.data','rb') as file:
    x_test, y_test = pickle.load(file)
    x_valid, y_valid = pickle.load(file)
    x_train,y_train = pickle.load(file)
    file.close()
print(x_train.shape)
print(y_train.shape)

print('Preprocessed the data..    ')
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
valid_generator = train_datagen.flow(x_valid, y_valid, batch_size=batch_size)
test_generator = train_datagen.flow(x_test, y_test, batch_size=batch_size)
print('building Model')
#model = Pattern_Recognion_Model(x_train[1], y_train)
print('Train data shape ', x_train.shape)
#model = Pattern_Recognion_Model3(x_train[1], y_train, Filters=[248,128,64,512])
model = ConvolutionalModel(x_train[1].shape)

print('Training the Model ')
print(model.summary())
#model.fit(x_train, y_train, validation_data=(x_valid, y_valid),epochs=5, batch_size=batch_size)

model.fit_generator(train_generator ,validation_data=valid_generator,
                    steps_per_epoch=len(x_test) / 8, epochs=2,validation_steps=len(x_valid)/8)
# #print(val)

model_json = model.to_json()
with open("model/Basicmodel1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model/Basicmodel1.h5")
print("Saved model to disk")
#print('test Data results    ', model.evaluate_generator(test_generator, steps=len(x_test)/8))
#print('Validation Results   ', model.evaluate_generator(valid_generator, steps=len(x_valid)/8))


