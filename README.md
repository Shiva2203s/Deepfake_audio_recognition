# Deepfake_audio_recognition
"Deepfake Audio Detection using CNN, RNN, LSTM with Mel-Spectrograms"

# 🎙️ Deepfake Audio Detection using CNN + RNN + LSTM

This project aims to detect **deepfake audio** by classifying samples into three categories:  
✅ Real  
❌ Fake (AI-generated)  
⚠️ Hybrid (a mix of real and fake audio)

---

## 🔧 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **CNN + RNN + LSTM**
- **Mel-Spectrogram (feature extraction)**
- **Librosa, NumPy, Pydub**
- **Focal Loss for imbalanced data**

---

## 🧠 Model Architecture

> A hybrid architecture using:
- **CNN layers** to learn spatial features from Mel-spectrograms
- **RNN + LSTM layers** for temporal sequence modeling
- **Softmax** output for multi-class classification (Real / Fake / Hybrid)

---

## 📁 Dataset

- Collected real audio and fake (TTS/AI-generated) samples
- Hybrid audio created using **Pydub** to combine real and fake clips
- Converted all samples to Mel-Spectrograms for input into the model

---

## 🏋️‍♂️ Training

- Applied **focal loss** to reduce the impact of class imbalance
- Used **ModelCheckpoint** to save the best model
- Trained with **batch normalization**, **dropout**, and **Adam optimizer**

---

## 📊 Results

- Achieved high precision on unseen samples
- Effective in distinguishing subtle manipulations in hybrid audio
- Suitable for applications in **security**, **authentication**, and **media verification**

---

## ▶️ How to Run

1. Clone the repository  

2. Install requirements  

3. Run training  

4. Evaluate on test data  

---

## 📸 Sample Mel-Spectrogram

![Mel Spectrogram Example](images/mel_spectrogram_sample.png)

---

## 👨‍💻 Author

**Shivam Yadav**  
[LinkedIn](https://www.linkedin.com/in/shivam-yadav-3238a5229)  
Email: shivamyadav22sep@gmail.com

---

## 📄 License

This project is open-source for educational and research purposes.


