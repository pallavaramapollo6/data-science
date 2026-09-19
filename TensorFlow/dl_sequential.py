import tensorflow as tf

# Load the MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0–255 to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create an AI neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Configure the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the AI
model.fit(x_train, y_train, epochs=5)

# Save this model
model.save("mnist_model.keras")

# Test the AI
loss, accuracy = model.evaluate(x_test, y_test)

print("Test accuracy:", accuracy)

# Make a prediction
prediction = model.predict(x_test[:1])
print("Predicted digit:", prediction.argmax())
print("Actual digit:", y_test[0])
