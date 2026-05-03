import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("Rebuilding CNN architecture...")

base_model = MobileNetV2(
    weights=None,              # IMPORTANT
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(4, activation="softmax")
])

print("Loading trained weights...")

model.load_weights("cnn.weights.h5")

print("CNN model loaded successfully!")