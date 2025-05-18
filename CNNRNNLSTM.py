import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                    Dropout, BatchNormalization, Input, Reshape, LSTM)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                      ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Constants
FIXED_SHAPE = (128, 128)  # Spectrogram dimensions
BALANCED_DATA_PATH = "BALANCED_DATASET"  # Path to balanced dataset
CLASS_NAMES = ["Real", "Fake", "Hybrid"]

# 1. Enhanced Feature Extraction
def extract_features(file_path, sr=22050, fixed_shape=FIXED_SHAPE):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        # Pre-processing
        y = librosa.effects.preemphasis(y)
        y = librosa.util.normalize(y)
        
        # Feature extraction
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=fixed_shape[0], fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Fixed shape
        if S_db.shape[1] < fixed_shape[1]:
            S_db = np.pad(S_db, ((0, 0), (0, fixed_shape[1] - S_db.shape[1])), mode='constant')
        else:
            S_db = S_db[:, :fixed_shape[1]]
            
        return S_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. Load Balanced Dataset
def load_balanced_dataset(dataset_path):
    labels = {
        "Real": 0,
        "Fake": 1, 
        "Hybrid": 2
    }
    X, y = [], []
    
    for class_name, class_idx in labels.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Missing class folder: {class_path}")
            
        print(f"\nLoading {class_name} samples...")
        for file in tqdm(os.listdir(class_path)):
            if not file.lower().endswith((".wav", ".mp3", ".flac")):
                continue
                
            file_path = os.path.join(class_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(class_idx)
    
    return np.array(X), np.array(y)

# 3. Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0-1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1-y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss_fn

# 4. Load and Prepare Data
print("Loading balanced dataset...")
X, y = load_balanced_dataset(BALANCED_DATA_PATH)

# Verify balance
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for cls, count in zip(unique, counts):
    print(f"{CLASS_NAMES[cls]}: {count} samples")

# Preprocessing
X = X[..., np.newaxis]  # Add channel dimension
X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0,1]
y = keras.utils.to_categorical(y, num_classes=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Build Enhanced CNN + RNN + LSTM Model
def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # CNN feature extractor
    x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)

    # Reshape for RNN
    shape = x.shape
    x = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)  # (batch_size, timesteps, features)

    # RNN + LSTM layers
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)

    # Fully connected head
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.6)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

# Build model
model = build_model((FIXED_SHAPE[0], FIXED_SHAPE[1], 1))

# 6. Compile with Focal Loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(),
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# 7. Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=15, mode='max', 
                 restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_crl_model.h5", monitor='val_auc', 
                   mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                     patience=5, min_lr=1e-6, verbose=1)
]

# 8. Train Model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 9. Save Final Model
model.save("final_crl_model.h5")
print("✅ Model saved successfully!")

# 10. Evaluation
def plot_history(history):
    plt.figure(figsize=(15,5))
    
    # Accuracy
    plt.subplot(1,3,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1,3,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Focal Loss')
    plt.legend()
    
    # AUC
    plt.subplot(1,3,3)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
