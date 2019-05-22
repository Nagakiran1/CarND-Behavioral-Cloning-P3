
# Importing Required libraries
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from IPython.core.debugger import set_trace


def random_flip(image, steering_angle):
    """
    Function to Flip the images Horizontally randomly
    Flipped images will be assigned with the Negative steering angle mentioning the network in which direction curve associated with.
   
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_shadow(image):
    """
    Function to add shadow in images randomly at random places, Random shadows meant to make the Convolution model learn 
    Lanes and lane curvature patterns effectively in dissimilar places.
    """
    if np.random.rand() < 0.5:
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = image.shape[1] * np.random.rand(), 0
        x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
        xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    else:
        return image


def random_brightness(image):
    """
    Function to add random brightness to random images. Random brightness will allow the model to learn and coverge on lane lines 
    and lane curvatures even in low light conditions.
 
    """
    if np.random.rand() < 0.5:
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        image = np.array(image, dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        return image


    

def Image_Preprocessing(Drivinglog):
    '''
        Function to read each image in a directory and making them as a generalized array to feed Neural Network.
        Preprocessing of images included with 5 steps 
            1. Reading all training images and the driving log csv to create Train features and predicting variable data.
            2. Translating the images with the use Random flipping and assigning a steering angle negatively for each flipped image.
            3. Randomly adding shadow to each image 
            4. Concatenating all augmented data
            5. Splitting the data to train, test and valid randomly
    
    '''
    #1. Reading all training images and the driving log csv to create Train features and predicting variable data.
    CenterImages = np.concatenate([cv2.cvtColor(cv2.imread(load_dir +file), cv2.COLOR_BGR2RGB)[np.newaxis] for file in Drivinglog[Drivinglog.columns[0]].tolist()])
#     a = Drivinglog[Drivinglog.columns[3]].astype(float)
#     y_data = (a + ((((~a.between(-0.05,0.01))  & (a.gt(0.01))).astype(int) * 0.05) + (((~a.between(-0.05,0.01))  & (a.lt(-0.01))).astype(int) * -0.05))).values[:,None]
    y_data = Drivinglog[Drivinglog.columns[3]].astype(float).values[:,None]
    
    #FlipedCenterImages = np.concatenate([cv2.flip(image,1)[np.newaxis] for image in CenterImages])
    print('Translating the Image data  ... ')
    # 2. Translating the images with the use Random flipping and assigning a steering angle negatively for each flipped image.  
    val = [random_flip(image,steering_angle) for image, steering_angle in zip(CenterImages,y_data)]    
    TranslatedImages = np.concatenate([i[0][np.newaxis] for i in val])
    TranslatedSangle = np.array([i[1] for i in val])
    print('Adding shadow to image data. ...')
    
    
    # 3. Randomly adding shadow to each image 
    RBrightnesImages = np.concatenate([random_brightness(image)[np.newaxis] for image in CenterImages])
    print('Changing Brightness of image data ... ')
    
    # 4. Concatenating all augmented data
    X_data = np.concatenate((CenterImages, TranslatedImages, RBrightnesImages))
    y_data = np.concatenate((y_data, TranslatedSangle, y_data))
#     X_data = CenterImages #RBrightnesImages
#     y_data = y_data
    
    # 5. Splitting the data to train, test and valid randomly
    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    print('Train data shape ', X_data.shape)
    
    with open('/opt/ProcessedData.data','wb') as file:
        pickle.dump([x_test,y_test],file)
        pickle.dump([x_valid,y_valid],file)
        pickle.dump([x_train,y_train],file)
        file.close()
 
load_dir = '../Data/SampleData/data/'
Drivinglog = pd.read_csv(load_dir + 'driving_log.csv')
Drivinglog = pd.concat([Drivinglog[Drivinglog.steering.ne(0)],Drivinglog[Drivinglog.steering.between(-0.01,0.01)][:600]]).reset_index(drop=True)
Drivinglog = Drivinglog.sample(100)
Image_Preprocessing(Drivinglog)

        