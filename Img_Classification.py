# Image classification
#---------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import cv2
import h5py
import numpy as np
import mahotas as mt


imageClass = 125
blocks = 16
trainImg_path = "./database/trainImage"
data          = './result/data'
labels        = './result/label'
fixed_SE_size = tuple((600, 600))


# Texture
def texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mt.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram
def histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [blocks, blocks, blocks], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist
    return hist.flatten()

# Moment
def moment(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# get labels
Img_tr_label = os.listdir(trainImg_path)

# sort labels
Img_tr_label.sort()
print(Img_tr_label)


labels_Image = []
global_ft    = []

# loop training image 
for train_name in Img_tr_label:
    dir = os.path.join(trainImg_path, train_name)

    current_label = train_name

    for Img in range(1,imageClass+1):
        # Make file name
        file_dr = dir + "./" + str(Img) + ".jpg"

         img = cv2.imread(file_dr)
         img_resize = cv2.resize(img, fixed_size)

        # Step1: Global Feature extraction
        gf_moments = moment(img_resize)
        gf_haralick   = texture(img_resize)
        gf_histogram  = histogram(img_resize)

        # Step2: Concatenate global features
        cgf = np.hstack([gf_histogram, gf_haralick, gf_moments])

        global_ft.append(cgf)
        labels_Image.append(current_label)


# get the overall training label size
print("lable of training image {}".format(np.array(labels_Image).shape))


# get vector size
print("vector size {}".format(np.array(global_ft).shape))


# encode labels
leble          = LabelEncoder()
Names = np.unique(labels_Image)
target      = leble.fit_transform(labels_Image)

print("target label & shape : {} & {}".format(target.shape) , .format(target))

# scale features
scale_ft = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scale_ft.fit_transform(global_ft)

h5_lb = h5py.File(labels, 'w')
h5_dt = h5py.File(data, 'w')
h5_dt.create_dataset('dataset_target_Img', data=np.array(rescaled_features))
h5_lb.create_dataset('dataset_target_Img', data=np.array(target))

h5_lb.close()
h5_dt.close()

