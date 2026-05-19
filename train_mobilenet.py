import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


LABEL_FILE = "labels_binary.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_composition_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "mobilenet_label_classes.npy")

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(LABEL_FILE)

# Pastikan path pakai format string
df["image_path"] = df["image_path"].astype(str)
df["label"] = df["label"].astype(str)

# Encode label
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

np.save(ENCODER_PATH, label_encoder.classes_)

print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label_encoded"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_encoded"]
)

print("\nJumlah data:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze base model dulu agar training lebih ringan
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nMulai training tahap 1: feature extraction...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tuning ringan: buka sebagian layer terakhir MobileNetV2
print("\nMulai fine-tuning ringan...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

print("\nEvaluasi pada test set...")

best_model = tf.keras.models.load_model(MODEL_PATH)

test_generator.reset()
pred_prob = best_model.predict(test_generator)
pred_class = (pred_prob > 0.5).astype(int).reshape(-1)

true_class = test_generator.classes

accuracy = accuracy_score(true_class, pred_class)

print("\nAccuracy:", accuracy)

# class_indices dari generator mungkin urut alfabetis
print("\nClass indices generator:")
print(test_generator.class_indices)

target_names = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_class, pred_class, target_names=target_names))

print("Confusion Matrix:")
print(confusion_matrix(true_class, pred_class))

print("\nModel MobileNetV2 berhasil disimpan di:")
print(MODEL_PATH)
print("Label classes disimpan di:")
print(ENCODER_PATH)