# model_training/model_train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os, json

# -----------------------------
# 1. Dataset Paths
# -----------------------------
DATA_DIR = "dataset"

if os.path.exists(os.path.join(DATA_DIR, "Indian-monuments", "images")):
    TRAIN_DIR = os.path.join(DATA_DIR, "Indian-monuments", "images", "train")
    VAL_DIR = os.path.join(DATA_DIR, "Indian-monuments", "images", "test")
elif os.path.exists(os.path.join(DATA_DIR, "train")):
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "test")
else:
    raise FileNotFoundError("Train/test directories not found.")

print(f"Training Directory → {TRAIN_DIR}")
print(f"Validation Directory → {VAL_DIR}")

# -----------------------------
# 2. Your FULL SITE LIST
# -----------------------------
TARGET_SITES = [
    'Taj Mahal', 'Qutub Minar', 'Gateway of India', 'Ajanta Caves', 'Alai Darwaza',
    'Alai Minar', 'Basilica of Bom Jesus', 'Charar-E-Sharif', 'Charminar', 'Chhota Imambara',
    'Ellora Caves', 'Fatehpur Sikri', 'Golden Temple', 'Hawa Mahal', "Humayun's Tomb",
    'Iron Pillar', 'Jamali Kamali Tomb', 'Khajuraho', 'Lotus Temple', 'Mysore Palace',
    'Sun Temple Konark', 'Thanjavur Temple', 'Victoria Memorial'
]

# Normalize folder names for matching
def normalize(name):
    return name.lower().replace(" ", "").replace("'", "").replace("-", "").replace("_", "")

# -----------------------------
# 3. Match site names with folder names
# -----------------------------
train_folders = os.listdir(TRAIN_DIR)
folder_map_train = {}

for site in TARGET_SITES:
    site_norm = normalize(site)
    match = None

    for folder in train_folders:
        if normalize(folder).startswith(site_norm[:5]):  # first 5 letters match
            match = folder
            break

    if match:
        folder_map_train[site] = match
    else:
        print(f"⚠️ WARNING: No folder found in TRAIN for → {site}")

print("\nTRAIN folder mapping:")
print(folder_map_train)

# Validation folders
test_folders = os.listdir(VAL_DIR)
folder_map_test = {}

for site in TARGET_SITES:
    site_norm = normalize(site)
    match = None

    for folder in test_folders:
        if normalize(folder).startswith(site_norm[:5]):
            match = folder
            break

    if match:
        folder_map_test[site] = match
    else:
        print(f"⚠️ WARNING: No folder found in TEST for → {site}")

print("\nTEST folder mapping:")
print(folder_map_test)

# Remove missing ones
TRAIN_CLASSES = list(folder_map_train.values())
TEST_CLASSES = list(folder_map_test.values())

print("\nFinal classes used for training:")
print(TRAIN_CLASSES)

# -----------------------------
# 4. Image Generators
# -----------------------------
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(
    TRAIN_DIR,
    classes=TRAIN_CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

val_flow = val_gen.flow_from_directory(
    VAL_DIR,
    classes=TEST_CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

NUM_CLASSES = train_flow.num_classes
print(f"\nDetected {NUM_CLASSES} classes.")

# -----------------------------
# 5. Build Model
# -----------------------------
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 6. Train
# -----------------------------
callbacks = [
    ModelCheckpoint("model_best.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
]

model.fit(train_flow, epochs=EPOCHS, validation_data=val_flow, callbacks=callbacks)

# -----------------------------
# 7. Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)
model.save("model/heritage_model.h5")

with open("model/class_indices.json", "w") as f:
    json.dump(train_flow.class_indices, f, indent=2)

print("\nTraining Completed Successfully!")
print("Model saved at: model/heritage_model.h5")
