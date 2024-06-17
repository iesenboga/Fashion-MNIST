import openml
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

# Load dataset
dataset = openml.datasets.get_dataset(40996)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Summary
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
print(f"Unique classes: {np.unique(y)}")
df = pd.DataFrame(X)
df['label'] = y
print(df.head())
print(df.describe())

# Dimensionality reduction
X_flat = X.to_numpy().reshape(X.shape[0], -1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Shape of PCA-transformed data: {X_pca.shape}")

# Visualization
le = LabelEncoder()
y_encoded = le.fit_transform(y)
colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y))))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[i] for i in y_encoded], s=1)
plt.colorbar()
plt.title('PCA of Fashion-MNIST')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# VGG-16 Embeddings
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block5_pool').output)
X_resized = np.array([np.resize(img, (32, 32, 3)) for img in X.to_numpy().reshape(-1, 28, 28)])
X_preprocessed = preprocess_input(X_resized)
X_features = model.predict(X_preprocessed)
X_flat_features = X_features.reshape(X_features.shape[0], -1)

# Perform PCA on VGG-16 features
pca_vgg = PCA(n_components=2)
X_pca_vgg = pca_vgg.fit_transform(X_flat_features)
plt.scatter(X_pca_vgg[:, 0], X_pca_vgg[:, 1], c=[colors[i] for i in y_encoded], s=1)
plt.colorbar()
plt.title('PCA of VGG-16 Embeddings for Fashion-MNIST')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Clustering on VGG-16 embeddings
kmeans_vgg = KMeans(n_clusters=10, random_state=42)
clusters_vgg = kmeans_vgg.fit_predict(X_pca_vgg)
conf_matrix_vgg = confusion_matrix(y_encoded, clusters_vgg)
conf_matrix_normalized_vgg = conf_matrix_vgg.astype('float') / conf_matrix_vgg.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_normalized_vgg, annot=True, fmt='.2f', cmap='viridis')
plt.title('Normalized Confusion Matrix for K-Means Clustering on VGG-16 Embeddings')
plt.xlabel('Cluster Label')
plt.ylabel('True Label')
plt.show()

# Classification on VGG-16 embeddings
X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg = train_test_split(X_flat_features, y_encoded, test_size=0.2, random_state=42)
clf_vgg = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_vgg.fit(X_train_vgg, y_train_vgg)
y_pred_vgg = clf_vgg.predict(X_test_vgg)
print(classification_report(y_test_vgg, y_pred_vgg))
