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

# Save the predicted and actual counts for each test sample
results = []
for i in range(len(y_test)):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    actual_count = int(y_test[i])
    predicted_count = int(round(y_pred[i]))
    results.append((img_path, predicted_count, actual_count))

# Sort the results by the difference between actual count and predicted count
results.sort(key=lambda x: abs(x[2] - x[1]))

# Select the best and worst results based on prediction accuracy
best_result = results[0]
worst_result = results[-1]

# Calculate the prediction accuracy for the best and worst results
best_accuracy = (1 - abs(best_result[1] - best_result[2]) / best_result[1]) * 100
worst_accuracy = (1 - abs(worst_result[2] - worst_result[1]) / worst_result[2]) * 100

# Display the best and worst results
plt.figure(figsize=(10, 5))

# Display the best result
plt.subplot(1, 2, 1)
img = cv2.imread(best_result[0], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title('Best score\nActual count: {}\nPredicted count: {}\nAccuracy: {:.2f}%'.format(
    best_result[2], best_result[1], best_accuracy))

# Display the worst result
plt.subplot(1, 2, 2)
img = cv2.imread(worst_result[0], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title('Worst score\nActual count: {}\nPredicted count: {}\nAccuracy: {:.2f}%'.format(
    worst_result[2], worst_result[1], worst_accuracy))

plt.show()
