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

print('defining the GLCM properties')
# Define the GLCM properties
dists = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']


print("Extract the GLCM features from the images")
# Extract the GLCM features from the images
glcm_features = []
for i in range(image_features.shape[0]):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    glcm = greycomatrix(img, distances=dists, angles=angles, symmetric=True, normed=True)
    glcm_props = np.hstack([greycoprops(glcm, prop).ravel() for prop in props])
    glcm_features.append(glcm_props)
glcm_features = np.vstack(glcm_features)

print('Split the dataset into training and testing sets')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(glcm_features, labels, test_size=0.2, random_state=42)

# Train SVR
print('Train SVR')
svr = SVR(kernel='rbf', epsilon=0.1, C=100)
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error: {:.2f}'.format(mae))

# Save the trained model
dump(svr, 'svr_model.joblib')

# Display an image from the test set with the predicted and actual counts
img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_000001.jpg')
img = cv2.imread(img_path)
actual_count = labels_df[labels_df['id'] == 1]['count'].values
predicted_count = svr.predict(X_test[0].reshape(1,-1))[0]
ax = plt.subplot(1, 1, 1)
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.axis('off')
ax.text(10, 30, 'Actual count: {}'.format(actual_count[0]), color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
ax.text(10, 60, 'Predicted count: {:.2f}'.format(predicted_count), color='white', fontsize=10, bbox=dict(facecolor='blue', alpha=0.5))
plt.show()