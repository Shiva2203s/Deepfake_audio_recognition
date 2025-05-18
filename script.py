import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image

# Set Streamlit page config FIRST
st.set_page_config(page_title="Deepfake Audio Analyzer", layout="wide")

# Constants
FIXED_SHAPE = (128, 128)
MODEL_PATH = "best_crl_model.h5"
CLASS_NAMES = ["Real", "Fake", "Hybrid"]
CLASS_COLORS = ["🟢", "🔴", "🟡"]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_model()

def extract_features(file_path, sr=22050, fixed_shape=FIXED_SHAPE):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        y = librosa.effects.preemphasis(y)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=fixed_shape[0], fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        if S_db.shape[1] < fixed_shape[1]:
            S_db = np.pad(S_db, ((0, 0), (0, fixed_shape[1] - S_db.shape[1])), mode='constant')
        else:
            S_db = S_db[:, :fixed_shape[1]]

        S_db = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))

        return S_db, y, sr
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

# UI layout
st.title("🔊 Deepfake Audio Detection (3-Class)")
st.markdown("""
    Upload an audio file to classify as **Real**, **Fake**, or **Hybrid**.  
    Supported formats: WAV, MP3, FLAC
""")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path)

    with st.spinner("Analyzing audio..."):
        features, y, sr = extract_features(temp_path)

        if features is not None:
            input_data = np.expand_dims(features[..., np.newaxis], axis=0)
            predictions = model.predict(input_data)[0]
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎙️ Prediction")
                st.markdown(f"""
                **Class:** {CLASS_COLORS[predicted_class]} {CLASS_NAMES[predicted_class]}  
                **Confidence:** {confidence:.1%}
                """)

                # Horizontal bar chart
                fig_bar, ax = plt.subplots(figsize=(8, 3))
                bars = ax.barh(CLASS_NAMES, predictions, color=['green', 'red', 'gold'])
                ax.set_xlim(0, 1)
                ax.bar_label(bars, fmt='%.2f')
                ax.set_title("Class Probabilities")
                st.pyplot(fig_bar)

                # Pie chart for hybrid breakdown
                if CLASS_NAMES[predicted_class] == "Hybrid":
                    st.subheader("🔬 Hybrid Breakdown (based on model confidence)")
                    labels = ["Real", "Fake", "Hybrid"]
                    values = predictions
                    colors = ['green', 'red', 'gold']

                    fig_pie, ax = plt.subplots()
                    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig_pie)

                    st.markdown("""
                    > This chart shows the model’s confidence distribution for Hybrid audio.  
                    > Not an exact mix percentage, but gives insight into how much it leans toward Real or Fake.
                    """)

            with col2:
                st.subheader("📊 Audio Analysis")

                fig_wave, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set_title("Waveform")
                st.pyplot(fig_wave)

                fig_spec, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(features, x_axis='time', y_axis='mel',
                                               sr=sr, fmax=8000, ax=ax, cmap='magma')
                fig_spec.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title("Mel Spectrogram")
                st.pyplot(fig_spec)

            st.markdown

            os.remove(temp_path)
        else:
            st.error("Failed to process the audio file. Please try another file.")

   

# Footer
st.markdown("---")
st.markdown("""
*Model trained on balanced dataset with:  
- 3-class CNN classifier  
- Focal loss for imbalance handling  
- Mel spectrogram features*
""")
