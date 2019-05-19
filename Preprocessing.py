import os
import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from IPython.core.debugger import set_trace

test_ratio = 0.2
valid_ratio = 0.2

FolderPath = 'data/'
Driving_log = pd.read_csv(FolderPath+'driving_log.csv',header=None)
Driving_log[0] = Driving_log[0].str.rsplit('/',1).str[1]
Driving_log[1] = Driving_log[1].str.rsplit('/',1).str[1]
Driving_log[2] = Driving_log[2].str.rsplit('/',1).str[1]
                                  
files = os.listdir(FolderPath+'IMG/')
center_images_saved = [file for file in files if file.startswith('center')]
Driving_log = Driving_log.loc[Driving_log[0].isin(center_images_saved)].iloc[:100]
print('Processed images', Driving_log.shape)
print(Driving_log.describe().T)
print(Driving_log.shape)

def random_translate(image, steering_angle, range_x=70, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = image.shape[1] * np.random.rand(), 0
    x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
    xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    if np.random.rand() < 0.5:
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        return image

def PreprocessImages(Driving_log,FolderPath):
    print('Reading Image data')
    CenterImages = np.concatenate([cv2.cvtColor(cv2.imread(FolderPath+file),cv2.COLOR_BGR2RGB)[np.newaxis] for file in Driving_log[Driving_log.columns[0]].tolist()])
#     LeftImages = np.concatenate([cv2.imread(FolderPath+file)[np.newaxis] for file in Driving_log[Driving_log.columns[1]].tolist()])
#     RightImages = np.concatenate([cv2.imread(FolderPath+file)[np.newaxis] for file in Driving_log[Driving_log.columns[2]].tolist()])
    
    y_data = Driving_log[Driving_log.columns[3]].astype(np.float64).values[:,None]
    
    FlipedCenterImages = np.concatenate([cv2.flip(image,1)[np.newaxis] for image in CenterImages])
    
    #CenterImages = CenterImages[:,70:-25,:,:]
    #LeftImages = LeftImages[:,70:-25,:,:]
    #RightImages = RightImages[:,70:-25,:,:]

    #CenterImages = CenterImages/255 - 0.5
    #LeftImages = LeftImages/255 - 0.5
    #RightImages = RightImages/255 - 0.5

    #X_data = np.concatenate((LeftImages,CenterImages,RightImages),axis=1)
    X_data = CenterImages
    print('Translating the Image data  ... ')
    val = [random_translate(image,steering_angle) for image, steering_angle in zip(CenterImages,y_data)]    
    TranslatedImages = np.concatenate([i[0][np.newaxis] for i in val])
    TranslatedSangle = np.array([i[1] for i in val])
    
    print('Adding shadow to image data. ...')
    RShadowCenterImages = np.concatenate([random_shadow(image)[np.newaxis] for image in TranslatedImages])
    
    print('Changing Brightness of image data ... ')
    RBrightnesImages = np.concatenate([random_brightness(image)[np.newaxis] for image in RShadowCenterImages])
    
#     X_data = np.concatenate((CenterImages, FlipedCenterImages,TranslatedImages, RShadowCenterImages,RBrightnesImages))
#     y_data = np.concatenate((y_data, -y_data, TranslatedSangle, y_data, y_data))
    X_data = np.concatenate((FlipedCenterImages, RBrightnesImages))
    y_data = np.concatenate((-y_data, TranslatedSangle))
    
    print('Splitting the data to train,test and valid data ')
    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_ratio, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=42)
    print(x_train.shape)
    return x_train, x_test, x_valid, y_train, y_test, y_valid
print('Preprocessing Data ...')
x_train, x_test, x_valid, y_train, y_test, y_valid = PreprocessImages(Driving_log,FolderPath+'IMG//')
print('Saving data ...')
with open('data/ProcessedData.data','wb') as file:
    pickle.dump([x_test,y_test],file)
    pickle.dump([x_valid,y_valid],file)
    pickle.dump([x_train,y_train],file)
    file.close()

print('\nFinished Preprocessing ')