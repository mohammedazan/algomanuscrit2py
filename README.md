# ✍️ Handwritten Algorithm to Python Code  
### Master Project – Deep Learning & Artificial Neural Networks

---

## 📌 Project Overview

This project aims to develop an **intelligent application** capable of converting  
**handwritten algorithms** (captured as images) into **executable Python code**.

It combines **Deep Learning**, **Computer Vision**, and **rule-based parsing** to bridge the gap between handwritten pseudo-code and real programming languages.

---

## 🎯 Objectives

- 🧠 Recognize handwritten algorithm text using Deep Learning (OCR)
- 🔍 Improve robustness against noisy images and handwriting variations
- 🧩 Parse algorithmic logic (loops, input/output, variables)
- 🐍 Generate valid and readable Python code
- 🌐 Provide a simple and interactive web interface

---

## 🏗️ System Architecture

### 🔁 Global Pipeline

```

┌─────────────┐
│  Web App    │  (Streamlit)
│ Upload Img  │
└──────┬──────┘
↓
┌─────────────┐
│ Preprocess  │  (OpenCV)
│ - Grayscale │
│ - Threshold │
│ - Resize    │
└──────┬──────┘
↓
┌─────────────┐
│ DL OCR      │  (CNN / CRNN + CTC)
│ Handwritten │
└──────┬──────┘
↓
┌─────────────┐
│ Text Parser │  (Rules / Mapping)
│ Algorithm   │
└──────┬──────┘
↓
┌─────────────┐
│ Python Code │
│ Generator   │
└─────────────┘

```

---

## 🧩 Project Structure

```

handwritten_algo_to_python/
│
├── data/
│   ├── images/                # Handwritten algorithm images
│   └── annotations/
│       ├── dataset.csv        # Dataset annotations (tabular)
│       └── dataset.json       # Dataset annotations (robust format)
│
├── src/
│   ├── preprocessing/
│   │   └── image_preprocess.py    # Image preprocessing (OpenCV)
│   │
│   ├── ocr/
│   │   ├── model.py               # OCR model architecture
│   │   ├── train.py               # Model training
│   │   └── predict.py             # OCR inference
│   │
│   ├── parser/
│   │   └── algo_to_python.py      # Algorithm → Python conversion
│   │
│   └── app/
│       └── app.py                 # Streamlit web application
│
├── notebooks/
│   └── exploration.ipynb          # Dataset & experiments
│
├── requirements.txt
└── README.md

````

---

## 📊 Dataset Description

- 📸 **Images**: Handwritten algorithms (multiple categories)
- 🏷️ **Labels**:
  - Algorithm pseudo-code (text)
  - Corresponding Python code (for evaluation)

### Supported Algorithm Types:
- 📥 Input / Output (Lire, Afficher)
- 🔁 Loops (For)
- ➕ Arithmetic operations
- 🔍 Simple conditions (future extension)

> ⚠️ Dataset includes **multiline text and code**, requiring robust parsing strategies.

---

## 🧠 Deep Learning Component

### ✨ OCR Model
- CNN or CRNN-based architecture
- Trained to recognize handwritten algorithm text
- Handles:
  - Variable handwriting styles
  - Imperfect lighting
  - Noise and distortions

### 🔧 Technologies:
- **TensorFlow / Keras**
- **CTC loss** (for sequence prediction)
- **Data preprocessing & augmentation**

---

## 🔍 Image Preprocessing (OpenCV)

Applied before OCR to improve recognition accuracy:

- 🖤 Grayscale conversion
- 🌫️ Gaussian Blur (noise reduction)
- ⚫ Adaptive Thresholding
- 📐 Image resizing to fixed input size

---

## 🧩 Algorithm Parsing & Code Generation

The recognized text is transformed using **rule-based parsing**:

| Algorithm Instruction | Python Equivalent |
|----------------------|-------------------|
| `Lire(x)`            | `x = int(input())` |
| `Afficher(x)`        | `print(x)` |
| `Pour i de 1 à n`    | `for i in range(1, n+1):` |

This ensures:
- ✔️ Correct syntax
- ✔️ Readable Python code
- ✔️ Educational clarity

---

## 🌐 Web Application

Built using **Streamlit**:

Features:
- 📤 Upload handwritten image
- 👀 Preview preprocessing results
- 🧠 OCR text output
- 🐍 Generated Python code display

---

## ⚙️ Technologies Used

| Category | Tools |
|--------|-------|
| Language | Python 🐍 |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Handling | Pandas, NumPy |
| Web Interface | Streamlit |
| Visualization | Matplotlib |

---

## 🚀 How to Run

### 1️⃣ Activate Virtual Environment
```bash
venv\Scripts\activate
````

### 2️⃣ Run Dataset Validation

```bash
python src/data/dataset_loader.py
```

### 3️⃣ Run Preprocessing Demo

```bash
python src/preprocessing/image_preprocess.py
```

### 4️⃣ Launch Web App

```bash
streamlit run src/app/app.py
```

---

## 📈 Future Improvements

* 🔤 Character-level OCR optimization
* 📚 Larger and more diverse dataset
* 🧠 Transformer-based OCR models
* 🌍 Multi-language algorithm support
* 🧪 Accuracy and performance benchmarking

---

## 🎓 Academic Context

* 📘 Master: Data Science / Artificial Neural Networks
* 🧪 Module: Deep Learning
* 🗓️ Duration: 12 days
* 👥 Team: Minimum 3 students

---

## ✅ Conclusion

This project demonstrates how **Deep Learning** can be applied to real-world educational problems by combining:

* Computer Vision
* Neural Networks
* Algorithmic reasoning
* Software engineering best practices

It emphasizes **clarity, robustness, and educational value** over unnecessary complexity.

---

👨‍🎓 *Master Project – Deep Learning & Artificial Neural Networks*


