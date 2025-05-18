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

📌 Prerequisites
Anaconda (installed)

Python 3.7 or higher (comes with Anaconda)

Required libraries: TensorFlow, Keras, NumPy, Librosa, Pydub, Matplotlib, etc.

🚀 Step-by-Step Instructions
1️⃣ Open Anaconda Prompt
Go to Start Menu → search and open Anaconda Prompt.

2️⃣ Navigate to the Project Directory
Type the path where your project is located. Example:

cd D:\ML_Projects\Deepfake_audio_recognition
3️⃣ Create a Virtual Environment (Optional but Recommended)

conda create -n deepfake_env python=3.8
conda activate deepfake_env

4️⃣ Install Required Libraries
Use pip or conda inside Anaconda Prompt:
pip install -r requirements.txt
If there's no requirements.txt, install manually:

pip install tensorflow keras numpy matplotlib librosa pydub scikit-learn

5️⃣ Run the Project
1st step:
python CNNRNNLSTM.py(for training the model)
2nd step:
python script.py(for testing real time audio)

6️⃣ (Optional) View Outputs
Spectrograms / Plots will open in a window or be saved to a folder like outputs/

Accuracy, loss, and model performance will print in the console

📂 Dataset Location
Make sure the audio dataset is placed in a folder like ./data/ or update the script to point to the correct location.
(Should me taken from kaggle)



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


