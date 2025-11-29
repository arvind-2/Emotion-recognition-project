# -------------------------------
# Install required packages (run if not installed)
# -------------------------------
# !pip install librosa tensorflow scikit-learn numpy matplotlib

# -------------------------------
# Import libraries
# -------------------------------
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Paths and parameters
# -------------------------------
dataset_path = 'RAVDESS/audio_speech_actors_01-24/'  # Change this to your dataset path
max_len = 174  # Max time steps for MFCC sequences
n_mfcc = 40    # Number of MFCC features

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features_lstm(file_path, max_len=max_len):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs = mfccs.T  # Shape: (time_steps, n_mfcc)

    # Pad or truncate sequences
    if len(mfccs) < max_len:
        pad_width = max_len - len(mfccs)
        mfccs = np.pad(mfccs, pad_width=((0,pad_width),(0,0)), mode='constant')
    else:
        mfccs = mfccs[:max_len, :]
    return mfccs

# -------------------------------
# Load dataset
# -------------------------------
features = []
labels = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = int(file.split("-")[2])  # RAVDESS emotion label
            data = extract_features_lstm(file_path)
            features.append(data)
            labels.append(emotion)

X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Build LSTM model
# -------------------------------
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_len, n_mfcc)))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -------------------------------
# Train model
# -------------------------------
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# -------------------------------
# Evaluate model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"LSTM Test Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Save the model
# -------------------------------
model.save("emotion_recognition_lstm.h5")

# -------------------------------
# Load the model
# -------------------------------
loaded_model = load_model("emotion_recognition_lstm.h5")

# -------------------------------
# Predict emotion
# -------------------------------
def predict_emotion(file_path, model, le):
    features = extract_features_lstm(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    predicted_emotion = le.inverse_transform([predicted_class])[0]
    return predicted_emotion

# Example prediction
test_file = 'RAVDESS/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav'
emotion = predict_emotion(test_file, loaded_model, le)
print(f"Predicted Emotion: {emotion}")

# -------------------------------
# Plot training history
# -------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
