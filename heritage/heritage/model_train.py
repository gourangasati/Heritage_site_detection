# model_training/train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import json

# Data directory paths - adjust based on your actual dataset structure
DATA_DIR = "dataset"
# Check if data is in nested structure or direct structure
if os.path.exists(os.path.join(DATA_DIR, "Indian-monuments", "images")):
    TRAIN_DIR = os.path.join(DATA_DIR, "Indian-monuments", "images", "train")
    VAL_DIR = os.path.join(DATA_DIR, "Indian-monuments", "images", "test")
elif os.path.exists(os.path.join(DATA_DIR, "train")):
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
else:
    raise FileNotFoundError(f"Could not find training data. Checked: {DATA_DIR}/train and {DATA_DIR}/Indian-monuments/images/train")

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10
NUM_CLASSES = None  # will be read from generator.class_indices

# Verify directories exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

# Define the specific classes to train on
# Note: Folder names may differ between train and test directories
# We'll maintain consistent ordering: tajmahal, qutub_minar, India Gate
TARGET_CLASSES_BASE = ['tajmahal', 'qutub_minar']

# Check for India Gate folder names (they differ between train and test)
# Train folder: "India gate pics", Test folder: "India_gate"
india_gate_train = None
india_gate_test = None

# Find India Gate folder in train directory
train_folders = [f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))]
for folder in train_folders:
    if 'india' in folder.lower() and 'gate' in folder.lower():
        india_gate_train = folder
        break

# Find India Gate folder in test directory
test_folders = [f for f in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, f))]
for folder in test_folders:
    if 'india' in folder.lower() and 'gate' in folder.lower():
        india_gate_test = folder
        break

# Create class lists with consistent ordering
if india_gate_train and india_gate_test:
    # Maintain order: tajmahal, qutub_minar, India Gate
    TARGET_CLASSES_TRAIN = TARGET_CLASSES_BASE + [india_gate_train]
    TARGET_CLASSES_TEST = TARGET_CLASSES_BASE + [india_gate_test]
    print(f"Found India Gate folders: Train='{india_gate_train}', Test='{india_gate_test}'")
else:
    print("WARNING: Could not find India Gate folders in both train and test directories")
    TARGET_CLASSES_TRAIN = TARGET_CLASSES_BASE
    TARGET_CLASSES_TEST = TARGET_CLASSES_BASE

print(f"\nTraining on {len(TARGET_CLASSES_TRAIN)} classes:")
print(f"  Train classes: {TARGET_CLASSES_TRAIN}")
print(f"  Test classes: {TARGET_CLASSES_TEST}")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

# Filter to only include target classes
train_flow = train_gen.flow_from_directory(
    TRAIN_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode='categorical',
    classes=TARGET_CLASSES_TRAIN,  # Only use these classes
    shuffle=True
)

val_flow = val_gen.flow_from_directory(
    VAL_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode='categorical',
    classes=TARGET_CLASSES_TEST,  # Use test folder names
    shuffle=False
)

NUM_CLASSES = train_flow.num_classes
print(f"Number of classes: {NUM_CLASSES}")
print(f"Training classes: {train_flow.class_indices}")
print(f"Validation classes: {val_flow.class_indices}")

# Verify that train and validation have the same classes
if train_flow.class_indices != val_flow.class_indices:
    print("WARNING: Training and validation class indices don't match!")
    print("This may cause issues. Make sure both directories have the same class folders.")

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # freeze for initial training

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Calculate steps per epoch
steps_per_epoch = len(train_flow)
validation_steps = len(val_flow)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

callbacks = [
    ModelCheckpoint("model_best.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("\n=== Starting initial training (frozen base) ===")
model.fit(
    train_flow, 
    validation_data=val_flow, 
    epochs=EPOCHS, 
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Optionally unfreeze some base layers and fine-tune
print("\n=== Starting fine-tuning (unfreezing top layers) ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

# Count trainable layers
trainable_count = sum([1 for layer in model.trainable_weights])
print(f"Trainable parameters: {trainable_count}")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Use different checkpoint for fine-tuning
fine_tune_callbacks = [
    ModelCheckpoint("model_best_finetuned.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

model.fit(
    train_flow, 
    validation_data=val_flow, 
    epochs=5, 
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# Create model directory if it doesn't exist
MODEL_DIR = os.path.join("..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save final model + class indices
MODEL_PATH = os.path.join(MODEL_DIR, "heritage_model.h5")
CLASS_PATH = os.path.join(MODEL_DIR, "class_indices.json")

print(f"\n=== Saving model to {MODEL_PATH} ===")
model.save(MODEL_PATH)
print(f"Model saved successfully!")

with open(CLASS_PATH, "w") as f:
    json.dump(train_flow.class_indices, f, indent=2)
print(f"Class indices saved to {CLASS_PATH}")
print(f"\nTraining completed! Model saved to: {MODEL_PATH}")
