# This is a sample Python script.
from PIL import Image
import numpy as np
import pandas as pd
import cv2
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

img = cv2.imread('/Users/pranavsekhar/PycharmProjects/Cloud_Detection_412/understanding_cloud_organization/test_images/0a3d2d8.jpg', 0)
kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Input', img)
#cv2.waitKey(0)
cv2.imshow('Dilation', img_dilation)
cv2.imwrite('Dilated.png', img_dilation)
#cv2.waitKey(0)
#exit()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # Press ⌘F8 to toggle the breakpoint.
    img = Image.open('/Users/pranavsekhar/PycharmProjects/Cloud_Detection_412/understanding_cloud_organization/test_images/0a0f81b.jpg')

    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = np.asarray(img)
    img = cv2.imread('/Users/pranavsekhar/PycharmProjects/Cloud_Detection_412/understanding_cloud_organization/test_images/0a0f81b.jpg', -1)
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    cv2.imwrite('shadows_out.png', result)
    cv2.imwrite('shadows_out_norm.png', result_norm)
    # <class 'numpy.ndarray'>
    print(type(numpydata))

    #  shape
    #print(numpydata.shape)
    fgbg = cv2.createBackgroundSubtractorMOG2(128, cv2.THRESH_BINARY, 1)
    masked_image = fgbg.apply(numpydata)
    #print(numpydata)
    df = pd.read_csv('./understanding_cloud_organization/train.csv')
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df['Image_id'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df.drop(['Image_Label'], axis=1, inplace=True)
    print(df['label'].value_counts())
    print(df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
