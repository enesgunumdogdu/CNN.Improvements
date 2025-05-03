import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Reshape data to fit model input
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
# Define a CNN model @TODO improving the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
# Save the model
model.save('mnist_model.h5')
# Load the model
from keras.models import load_model # type: ignore
loaded_model = load_model('mnist_model.h5')
# Evaluate the loaded model
loss, accuracy = loaded_model.evaluate(x_test, y_test)
print(f'Loaded model test loss: {loss}, Loaded model test accuracy: {accuracy}')
# Predict using the loaded model
predictions = loaded_model.predict(x_test)
# Print the first 5 predictions
for i in range(5):
    print(f'Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}')

# # Define a custom loss function
# def custom_loss(y_true, y_pred):
#     return K.mean(K.square(y_true - y_pred), axis=-1)
# # Compile the model with the custom loss function
# model.compile(optimizer=Adam(), loss=custom_loss, metrics=['accuracy'])
# # Train the model with the custom loss function
# model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# # Evaluate the model with the custom loss function
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Custom loss model test loss: {loss}, Custom loss model test accuracy: {accuracy}')
