"""

Benchmark method1, looking to use SVM method over the cloud images to successfully identify the different cloud patterns.
Author: Pranav Sekhar
Date: 14th March, 2023
"""
import re

import PIL
import cv2
import numpy as np
import os
import pickle
from sklearn import svm
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

# Path to dataset directory
dataset_dir = "/path/to/dataset"
df = pd.read_csv('./understanding_cloud_organization/train.csv')
df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
df['Image_id'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
df.dropna(axis=0, inplace=True)
df.reset_index(inplace=True)
df.drop(['Image_Label','index'], axis=1, inplace=True)
df = df.loc[df['label'] != 'Sugar']



def rle2mask(rle, imgshape):
    """
    Given the encoding pixels extracted from train.csv, this creates a mask to only display the pattern sub-image
    :param rle:
    :param imgshape:
    :return:
    """
    width = imgshape[0]
    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]
    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

datalabels = {'Fish' : 0, 'Flower':0, 'Gravel': 0}
train_count = 0
test_count = 0

for row in df.itertuples():
    image = row.Image_id
    encoding = row.EncodedPixels
    label = row.label
    datalabels[label] += 1
    #print(label)
    img = cv2.imread('./understanding_cloud_organization/train_images/'+ image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = rle2mask(encoding, img.shape)
    x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
    masked_img = x * img
    masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    masked_gray = crop(masked_gray)
    number_of_black_pix = np.sum(masked_gray == 0)
    total_pix = np.sum(masked_gray >= -1)
    if number_of_black_pix >total_pix//2:
        cv2.imwrite(f'./understanding_cloud_organization/noise/{label}/{image}', masked_gray)
    elif datalabels[label]%4 == 0:
        test_count += 1
        cv2.imwrite(f'./understanding_cloud_organization/test/{label}/{image}', masked_gray)
    else:
        train_count += 1
        cv2.imwrite(f'./understanding_cloud_organization/train/{label}/{image}', masked_gray)

print(train_count, test_count)
exit()


#keypoints, descriptors = sift.detectAndCompute(masked_img, None)
#sift_image = cv2.drawKeypoints(masked_gray, keypoints, masked_img)
#masked_img = crop(masked_img)
#print(masked_img.shape)

sift = cv2.SIFT_create()


def generate_imgs(pattern,df):
    df = df[df['label'] == pattern].sample(1500)
    descriptors = []
    for row in df.itertuples():
        encoding = row.EncodedPixels
        image = row.Image_id
        img = cv2.imread('./understanding_cloud_organization/train_images/' + image)
        mask = rle2mask(encoding, img.shape)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        masked_img = mask * img
        #masked_img = crop(mask * img)
        masked_img = cv2.cvtColor(cv2.resize(masked_img, (296, 264)), cv2.COLOR_RGB2GRAY)
        img_ = hog(masked_img, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(2, 2))
        #print(img_.shape)
        #eit()
        #print(descriptor)
        descriptors.append(img_)
    return descriptors
fish_images = generate_imgs('Fish', df)
flower_images = generate_imgs('Flower', df)
sugar_images = generate_imgs('Gravel', df)
#gravel_images = generate_imgs('Gravel', df)




fish_images = np.vstack(fish_images)
flower_images = np.vstack(flower_images)
sugar_images = np.vstack(sugar_images)
#gravel_images = np.vstack(gravel_images)


fish_labels = np.ones(len(fish_images))
flower_labels = np.zeros(len(flower_images))
sugar_labels = np.ones(len(sugar_images))*2
#gravel_labels = np.ones(len(gravel_images))*3


# Concatenate the cloud and non-cloud descriptors and labels
descriptors = np.vstack((fish_images, flower_images,sugar_images))
labels = np.hstack((fish_labels, flower_labels,sugar_labels))

#exit()
#print(descriptors)
x_train,x_test,y_train,y_test = train_test_split(descriptors,labels,test_size=0.2,random_state=42)
# Train an SVM model on the concatenated descriptors and labels
print('Running model')
svc = svm.SVC(kernel='poly',probability=True)
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
print("Accuracy score of model is ",accuracy_score(y_pred=y_pred,y_true=y_test)*100)
print("Confusion Matrix:", confusion_matrix(y_pred=y_pred,y_true=y_test))
# Save the trained SVM model as a .pkl file
#with open("svm_model.pkl", "wb") as f:
   # pickle.dump(svm_model, f)

def slideExtract(image, windowSize=(296, 264), channel="RGB", step=12):
    # Converting to grayscale
    if channel == "RGB":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channel == "BGR":
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel.lower() != "grayscale" or channel.lower() != "gray":
        raise Exception("Invalid channel type")

    # We'll store coords and features in these lists
    coords = []
    features = []

    hIm, wIm = image.shape[:2]

    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1, w2 in zip(range(0, wIm - windowSize[0], step), range(windowSize[0], wIm, step)):

        for h1, h2 in zip(range(0, hIm - windowSize[1], step), range(windowSize[1], hIm, step)):
            window = img[h1:h2, w1:w2]
            features_of_window = hog(window, orientations=9, pixels_per_cell=(16, 16),
                                     cells_per_block=(2, 2)
                                     )

            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)

    return (coords, np.asarray(features))

example_image = np.asarray(PIL.Image.open("./understanding_cloud_organization/train_images/0a1b596.jpg"))
coords,features = slideExtract(example_image,channel="RGB")

from sklearn.preprocessing import MinMaxScaler

class Heatmap():

    def __init__(self, original_image):
        # Mask attribute is the heatmap initialized with zeros
        self.mask = np.zeros(original_image.shape[:2])

    # Increase value of region function will add some heat to heatmap
    def incValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 30

    # Decrease value of region function will remove some heat from heatmap
    # We'll use this function if a region considered negative
    def decValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] - 30

    def compileHeatmap(self):
        # As you know,pixel values must be between 0 and 255 (uint8)
        # Now we'll scale our values between 0 and 255 and convert it to uint8

        # Scaling between 0 and 1
        scaler = MinMaxScaler()

        self.mask = scaler.fit_transform(self.mask)

        # Scaling between 0 and 255
        self.mask = np.asarray(self.mask * 255).astype(np.uint8)

        # Now we'll threshold our mask, if a value is higher than 170, it will be white else
        # it will be black
        self.mask = cv2.inRange(self.mask, 170, 255)

        return self.mask

def decide(prob_list):
    prob_list = list(map(float,re.sub(' +', ' ',str(prob_list[0]).strip('[|]')).strip().split(' ')))
    #print(prob_list)
    max_ = max(prob_list)
    if max_ > 0.6:
        return prob_list.index(max_), max_
    else:
        return -1,-1

def detect(image):
    # Extracting features and initalizing heatmap
    coords, features = slideExtract(image)
    fish_htmp = Heatmap(image)
    flower_htmp = Heatmap(image)
    sugar_htmp = Heatmap(image)
    #gravel_htmp = Heatmap(image)
    flags = {}
    op_features = {}
    for i in range(len(features)):
        # If region is positive then add some heat
        decision = svc.predict_proba([features[i]])
        idx,value = decide(decision)
        #print(idx,value, decision)
        if idx == -1:
            continue
        if idx not in flags:
            flags[idx] = value
        else:
            if flags[idx] >= value:
                continue
        flags[idx] = value
        op_features[idx] = coords[i]
        #print(flags)
    for idx in op_features:
        if idx == 1:
            flower_htmp.incValOfReg(op_features[idx])
            # Else remove some heat
        elif idx == 2:
            sugar_htmp.incValOfReg(op_features[idx])
        #elif idx ==3:
            #gravel_htmp.incValOfReg(op_features[idx])
        else:
            fish_htmp.incValOfReg(op_features[idx])
    # Compiling heatmap
    sugar_mask = sugar_htmp.compileHeatmap()
    flower_mask = flower_htmp.compileHeatmap()
    fish_mask = fish_htmp.compileHeatmap()
    #gravel_mask = gravel_htmp.compileHeatmap()

    fish_cont, _ = cv2.findContours(fish_mask, 1, 2)[:2]
    flower_cont, _ = cv2.findContours(flower_mask, 1, 2)[:2]
    sugar_cont, _ = cv2.findContours(sugar_mask, 1, 2)[:2]
    #gravel_cont, _ = cv2.findContours(gravel_mask, 1, 2)[:2]
    for fish in fish_cont:
        # If a contour is small don't consider it
        if cv2.contourArea(fish) < 70 * 70 or cv2.contourArea(fish)  > 700 * 700:
            continue

        (x, y, w, h) = cv2.boundingRect(fish)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    for flower in flower_cont:
        # If a contour is small don't consider it
        if cv2.contourArea(flower) < 70 * 70 or cv2.contourArea(flower) > 700 * 700:
            continue

        (x, y, w, h) = cv2.boundingRect(flower)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
    for sugar in sugar_cont:
        # If a contour is small don't consider it
        if cv2.contourArea(sugar) < 70 * 70 or cv2.contourArea(sugar) > 700 * 700:
            continue

        (x, y, w, h) = cv2.boundingRect(sugar)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
    """for gravel in gravel_cont:
        # If a contour is small don't consider it
        if cv2.contourArea(gravel) < 70 * 70 or cv2.contourArea(gravel) > 700 * 700:
            continue

        (x, y, w, h) = cv2.boundingRect(gravel)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)"""
    return image

detected = detect(np.asarray(PIL.Image.open("./understanding_cloud_organization/test_images/0a3d2d8.jpg")))
cv2.imshow('lol', detected)
cv2.waitKey(0)
#cv2.imshow('image', sift_image)
#v2.waitKey(0)

