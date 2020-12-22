import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

filepath = "DehnungVerschiebungMatrix.csv"
df = pd.read_csv(filepath)
h1 = np.array(df)
# print(df)


# Training set
x = h1[:, 0:6]
y = h1[:, 6:10]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.6, random_state=75)


# Create Keras model
tf.keras.backend.clear_session()
tf.random.set_seed(60)

model = keras.models.Sequential([

    keras.layers.Dense(1024, input_dim=6, activation='tanh'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(1024, activation='tanh'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(256, activation='tanh'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='tanh'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation='tanh'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.01),

    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dropout(0.05),

    keras.layers.Dense(4, activation="linear"),
], name="Larger_network", )

# Gradient descent algorithm
optimizer = keras.optimizers.Adam(lr=0.005, decay=5e-4)

model.compile(optimizer=optimizer, warm_start=False,
              loss='mean_absolute_error')

history = model.fit(train_x, train_y,
                    epochs=1250, batch_size=128,
                    validation_data=(test_x, test_y),
                    verbose=1)


plt.plot(history.history['loss'])
plt.xlabel("No. of Iterations")
plt.ylabel("J(Theta1 Theta0)/Cost")
plt.show()

predict = np.array([[-165.070258009183, 0.0408137033058864, -161.315124372419,
                     0.0398370454247239, -157.408731226767, 0.0389030402486873]])
print(model.predict(predict))
