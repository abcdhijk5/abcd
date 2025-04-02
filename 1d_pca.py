import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_images(folder, image_size, max_images=10):
    file_list = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    num_images = min(len(file_list), max_images)
    image_matrix = np.zeros((num_images, image_size**2))

    for i in range(num_images):
        img = cv2.imread(os.path.join(
            folder, file_list[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        image_matrix[i, :] = img.flatten()

    return image_matrix, file_list[:num_images]


# Paths to training images
train_dog_path = './train/dogs'
train_cat_path = './train/cats'
image_size = 64
num_components = 10

# Load cat and dog images
cat_images, cat_files = load_images(train_cat_path, image_size)
dog_images, dog_files = load_images(train_dog_path, image_size)

# Combine dataset
X = np.vstack((cat_images, dog_images))

# Apply PCA
pca = PCA(n_components=num_components)
score = pca.fit_transform(X)
X_pca = pca.inverse_transform(score)


def show_images(original, reconstructed, file_list, title_text, image_size):
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Display original images
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i].reshape(image_size, image_size), cmap='gray')
        plt.title(f'Original: {file_list[i]}', fontsize=8)
        plt.axis('off')

        # Display reconstructed images
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed[i].reshape(
            image_size, image_size), cmap='gray')
        plt.title('Reconstructed', fontsize=8)
        plt.axis('off')

    plt.suptitle(title_text)
    plt.show()


# Show results for Cats and Dogs
show_images(cat_images, X_pca[:len(cat_images)], cat_files,
            'Cats - Original vs Reconstructed', image_size)
show_images(dog_images, X_pca[len(cat_images):], dog_files,
            'Dogs - Original vs Reconstructed', image_size)
