
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,Conv1D,Conv2D,Activation,LSTM,MaxPooling2D,Flatten, Cropping2D, Lambda, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model,model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Set a couple flags for training - you can ignore these for now
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

# Loads in InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
# We can use smaller than the default 299x299x3 input for InceptionV3
# which will speed up training. Keras v2.0.9 supports down to 139x139x3
input_size = 139


# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(input_size,input_size,3))
# inception = ResNet50(weights='imagenet',include_top=False,input_shape=(input_size,input_size,3))

inception.layers.pop()

if freeze_flag == True:
    ## TODO: Iterate through the layers of the Inception model
    ##       loaded above and set all of them to have trainable = False
    for layer in inception.layers[:-3]:
        layer.trainable = False


def InceptionModel(x_train):
    X_input = Input(x_train.shape)

    
    X = Cropping2D(cropping=((60,25), (0,0)))(X_input)

    # Re-sizes the input with Kera's Lambda layer & attach to cifar_input
    X = Lambda(lambda image: tf.image.resize( 
        image, (input_size, input_size)))(X)

    # Feeds the re-sized input into Inception model
    inp = inception(X)


    model = GlobalAveragePooling2D(data_format=None)(inception.get_output_at(-1))
    model = Dense(240)(model)
    model = Dense(64)(model)
    predictions = Dense(1,activation='relu')(model)

    # Creates the model, assuming your final layer is named "predictions"
    model = Model(inputs=X_input, outputs=predictions)
    # Compile the model
    model.compile(optimizer='Adam', loss='mse', metrics=['mse'])

    # Check the summary of this new model to confirm the architecture
    model.summary()
    return model

load_dir = '/opt/'
with open(load_dir + 'train.pickle', 'rb') as f:
    [x_train, y_train] = pickle.load(f)
with open(load_dir + 'valid.pickle', 'rb') as f:
    [x_valid, y_valid] = pickle.load(f)
with open(load_dir + 'test.pickle', 'rb') as f:
    [x_test, y_test] = pickle.load(f)
print('Train ',x_train.shape)

# with open('/opt/ProcessedData.data','rb') as file:
#     x_test, y_test = pickle.load(file)
#     x_valid, y_valid = pickle.load(file)
#     x_train,y_train = pickle.load(file)
#     file.close()

    
model = InceptionModel(x_train[0])


model.fit(x_train,y_train,epochs=2,validation_data=(x_valid,y_valid))

# Saving the model
model_json = model.to_json()
with open("model/model1.json", "w") as json_file:
    json_file.write(model_json)
model.save('model/model1.h5')
