# Importig required Libaries
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Flatten, Cropping2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda
from keras.backend import tf as ktf
import csv
import pickle
import cv2
import numpy as np
from IPython.core.debugger import set_trace




        
        

train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                   samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-6, rotation_range=10, width_shift_range=0.2, 
                                   height_shift_range=0.2,shear_range=0, zoom_range=0.1, channel_shift_range=0., fill_mode='nearest', cval=0., 
                                   horizontal_flip=False,vertical_flip=False, rescale=None, preprocessing_function=None,  data_format="channels_last")



def PatternRecognitionModel(input_shape):
    '''
    Pattern Recognition model consists of 24 Sequential layers 
        6 Convolutional layers followed by relu activation layer.
        5 Fully connected layers followed bye relu activation layer.
    Convolutional layers plays important role in segregating all lane curve extractions and curvation associated information.
    Fully connected layers plays important role in reducing network size layer by layer in extracting curvature of lanes in taking the steering angle prediction
    
    Loss
    ----
    mean squared error loss is considered in optimizing the model performance
    Optimizer
    --------
    Adam optimizer is considered in changing the learning rate to converging Neural network in getting high performance in prediction
    default learning rate of 0.9 to 0.999 increment of optimization with the step of 0.001 is considered
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    
    '''
    # Model
    model = Sequential()
# Convolutional
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
#     Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))
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
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1),use_bias=True, 
                     kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Fully connected
    model.add(Flatten())
    model.add(Dense(units=1164))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
    return model


def Pattern_Recognion_Model_API(X_train,y_train):
    '''
    Pattern Recognition model with the use Functional API method consists of 24 Sequential layers 
        6 Convolutional layers followed by relu activation layer.
        5 Fully connected layers followed bye relu activation layer.
    Convolutional layers plays important role in segregating all lane curve extractions and curvation associated information.
    Fully connected layers plays important role in reducing network size layer by layer in extracting curvature of lanes in taking the steering angle prediction
    
    Loss
    ----
    mean squared error loss is considered in optimizing the model performance
    Optimizer
    --------
    Adam optimizer is considered in changing the learning rate to converging Neural network in getting high performance in prediction
    default learning rate of 0.9 to 0.999 increment of optimization with the step of 0.001 is considered
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    
    '''

    X_input = Input(shape=X_train.shape, name='img_in')
    X =  Cropping2D(cropping=((70, 25), (0, 0)))(X_input)
    #X = Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))(X)
    X = Lambda(lambda x: (x / 255.0) - 0.5)(X)
    X = Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=32,kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    # Fully connected
    X = Flatten()(X)
    # model.add(Dropout(0.35))
    X = Dense(units=1164)(X)
    X = Activation('relu')(X)
    X = Dense(units=100)(X)
    X = Activation('relu')(X)
    X = Dense(units=50)(X)
    X = Activation('relu')(X)
    X = Dense(units=10)(X)
    X = Activation('relu')(X)
    X = Dense(units=1)(X)
    model=Model(inputs=X_input, outputs=X, name='Convolve')
    model.compile(optimizer='adam',loss='mean_squared_error',metrics = ['mse'])
    return model


# Hyper parameters considered for model
batch_size = 10
epochs = 10
# Loading the preprocessed data 
with open('/opt/ProcessedData.data','rb') as file:
    x_test, y_test = pickle.load(file)
    x_valid, y_valid = pickle.load(file)
    x_train,y_train = pickle.load(file)
    file.close()

print('Train data shape',x_train.shape)
print('Train data shape ', x_train.shape)
    
# Initializing the Generator function    
datagen = ImageDataGenerator()

# Initializing the train, test and validation data to Augmentation data generator
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
valid_generator = datagen.flow(x_valid, y_valid, batch_size=batch_size)
test_generator = datagen.flow(x_test, y_test, batch_size=batch_size)

# Initializing the Convolutional model
model = PatternRecognitionModel(x_train[0].shape)
#model = PatternRecognitionModel_API(x_train[0].shape)
model.summary()

#Training  the model
model.fit_generator(train_generator, steps_per_epoch=int(np.ceil(len(x_train) / float(batch_size))), epochs=epochs, workers=4,
                    verbose=1, validation_data=valid_generator, validation_steps=int(np.ceil(len(x_valid) / float(batch_size))))

# Saving the model to disk
model_json = model.to_json()
with open("model/model_Pdata.json", "w") as json_file:
    json_file.write(model_json)

model.save('model/model_Pdata.h5')
print('Models saved Successfully')
