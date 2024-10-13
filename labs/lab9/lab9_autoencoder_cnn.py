import numpy as np
import matplotlib.pyplot as plt
import keras
# from keras import layers
# from keras.datasets import mnist
# from keras.models import Model

def preprocess(array):
    array = array.astype("float32")/255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def noise(array):
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2, labels):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    labels_sample = labels[indices]

    plt.figure(figsize=(20, 4))
    for i , (image1, image2, label) in enumerate(zip(images1, images2, labels_sample)):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(image1.reshape(28, 28), cmap="gray")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i+1+n)
        if label % 2 == 0:
            plt.imshow(image2.reshape(28, 28), cmap="Blues")  
        else:
            plt.imshow(image2.reshape(28, 28), cmap="Reds") 
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()

train_data = preprocess(train_data)
test_data = preprocess(test_data)

noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

display(train_data, noisy_train_data, train_labels)

input = keras.layers.Input(shape=(28,28,1))

# encoder
x = keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(input)
x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
x = keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

# decoder
x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = keras.layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

# autoencoder
autoencoder = keras.models.Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=1,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data)
)
predictions = autoencoder.predict(test_data)
display(test_data, predictions, test_labels)

autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=2,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_data, test_data)
)
predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions, test_labels)