import pandas as pd
import cv2
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

dataset_dir = "/path/to/dataset"
df = pd.read_csv('./understanding_cloud_organization/train.csv')
df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
df['Image_id'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
df.dropna(axis=0, inplace=True)
df.reset_index(inplace=True)
df.drop(['Image_Label','index'], axis=1, inplace=True)

orb = cv2.ORB_create()

images = list(df['Image_id'].unique())
def creat(pattern,df):
    descriptors = []
    df = df[df['label'] == pattern].sample(1500)
    images = list(df['Image_id'].unique())
    for image in images:
        img = cv2.imread('./understanding_cloud_organization/train_images/' + image)
        keypoints, descriptor = orb.detectAndCompute(img, None)
        descriptor = np.array(descriptor)[np.random.randint(descriptor.shape[0], size=40)]
        descriptors.append(descriptor)
    return descriptors

fish_images = creat('Fish', df)
flower_images = creat('Flower', df)
sugar_images = creat('Sugar', df)

print("YOLO")
fish_images = np.vstack(fish_images)
flower_images = np.vstack(flower_images)
sugar_images = np.vstack(sugar_images)


fish_labels = np.ones(len(fish_images))
flower_labels = np.zeros(len(flower_images))
sugar_labels = np.ones(len(sugar_images))*2


# Concatenate the cloud and non-cloud descriptors and labels
descriptors = np.vstack((fish_images, flower_images,sugar_images))
labels = np.hstack((fish_labels, flower_labels,sugar_labels))
# Create a vocabulary by clustering the descriptors
kmeans = KMeans(n_clusters=50)
kmeans.fit(np.vstack(descriptors))
vocabulary = kmeans.cluster_centers_
print("YAhoo")
# Convert each image into a bag of visual words
features = []
for descriptor in descriptors:
    histogram = np.zeros(len(vocabulary))
    for d in descriptor:
        distances = np.linalg.norm(vocabulary - d, axis=1)
        nearest_cluster = np.argmin(distances)
        histogram[nearest_cluster] += 1
    features.append(histogram)

print("Hawaii")
# Train the SVM classifier on the features and labels
x_train,x_test,y_train,y_test = train_test_split(descriptors,labels,test_size=0.2,random_state=42)
# Train an SVM model on the concatenated descriptors and labels
print('Running model')
svc = svm.SVC(kernel='poly',probability=True)
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
print("Accuracy score of model is ",accuracy_score(y_pred=y_pred,y_true=y_test)*100)

print("model")
# Load a new cloud image to classify
new_image = cv2.imread('./understanding_cloud_organization/test_images/0a03bc6.jpg')
keypoints, descriptor = orb.detectAndCompute(new_image, None)

# Convert the new image descriptor into a bag of visual words
histogram = np.zeros(len(vocabulary))
for d in descriptor:
    distances = np.linalg.norm(vocabulary - d, axis=1)
    nearest_cluster = np.argmin(distances)
    histogram[nearest_cluster] += 1

# Use the SVM classifier to predict the label of the new image
predicted_label = svm.predict([histogram])[0]

print("The predicted label of the new cloud image is:", predicted_label)