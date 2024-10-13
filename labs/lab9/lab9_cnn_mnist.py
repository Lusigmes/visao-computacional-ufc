import numpy as np
import keras
from keras import layers
import cv2

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train) ,  (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 2

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)

print("test loss", score[0])
print("test accurarcy", score[1])

# Salvando o modelo treinado
model.save("mnist_digit_recognition_model.h5")

# Prevendo novos dígitos
# Suponha que você tenha uma nova imagem (em escala de cinza, tamanho 28x28)
# Substitua 'new_image' pela sua imagem
# new_image = x_test[5]  # Exemplo: pegando a primeira imagem do conjunto de teste

new_image = cv2.imread('d.jpeg', cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(new_image, (28, 28))

# Normaliza a imagem (opcional)
new_image = new_image / 255.0

# Pré-processamento da nova imagem
new_image = np.expand_dims(new_image, axis=0)  # Adiciona a dimensão do batch

# Fazendo a predição
predicted_digit = np.argmax(model.predict(new_image), axis=-1)
print(f"Dígito previsto: {predicted_digit[0]}")