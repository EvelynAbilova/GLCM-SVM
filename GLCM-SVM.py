import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from skimage.feature import greycomatrix, greycoprops
from joblib import dump

# Define the path to the MALL dataset
MALL_DIR = 'MALL/'

# Load the labels
labels_df = pd.read_csv(os.path.join(MALL_DIR, 'labels.csv'))

# Load the pre-extracted image features
image_features = np.load(os.path.join(MALL_DIR, 'images.npy'))

# Load labels
labels = np.load(os.path.join(MALL_DIR, 'labels.npy'))

# Define the GLCM properties
dists = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Extract the GLCM features from the images
glcm_features = []
for i in range(image_features.shape[0]):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    glcm = greycomatrix(img, distances=dists, angles=angles, symmetric=True, normed=True)
    glcm_props = np.hstack([greycoprops(glcm, prop).ravel() for prop in props])
    glcm_features.append(glcm_props)
glcm_features = np.vstack(glcm_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(glcm_features, labels, test_size=0.2, random_state=42)

# Train SVR
svr = SVR(kernel='rbf', epsilon=0.1, C=100)
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error: {:.2f}'.format(mae))

# Save the trained model
dump(svr, 'svr_model.joblib')

# Sort the predictions in descending order
sorted_indices = np.argsort(y_pred)[::-1]

# Visualize the top 5 and bottom 5 predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

for i in range(5):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(sorted_indices[i] + 1))
    img = cv2.imread(img_path)
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].text(-0.2, 0.5, 'Best Score', fontsize=12, rotation=90, va='center', ha='center', transform=axes[0, 0].transAxes)
    axes[0, i].set_title('Prediction: {:.2f}'.format(y_pred[sorted_indices[i]]))
    axes[0, i].set_xlabel('Actual: {}'.format(int(labels[sorted_indices[i]])), fontsize=10)

    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(sorted_indices[-i - 1] + 1))
    img = cv2.imread(img_path)
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].text(-0.2, 0.5, 'Worth Score', fontsize=12, rotation=90, va='center', ha='center', transform=axes[1, 0].transAxes)
    axes[1, i].set_title('Prediction: {:.2f}'.format(y_pred[sorted_indices[-i - 1]]))
    axes[1, i].set_xlabel('Actual: {}'.format(int(labels[sorted_indices[-i - 1]])), fontsize=10)

plt.tight_layout()
plt.show()
