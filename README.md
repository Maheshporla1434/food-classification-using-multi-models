# 🍽️ Deep Learning Food Classification using Multiple Models

**Custom CNN • VGG16 • ResNet50 • Flask Deployment**

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Food%20Classification-orange" />
  <img src="https://img.shields.io/badge/Models-CNN%20%7C%20VGG16%20%7C%20ResNet50-blue" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-green" />
  <img src="https://img.shields.io/badge/Backend-Flask-lightgrey" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-red" />
</p>

---

## 🚀 Project Overview

This project focuses on building a **deep learning–based food image classification system** using three architectures:

* 🧠 **Custom CNN (built from scratch)**
* 🐥 **VGG16 Transfer Learning**
* 🏋️ **ResNet50 Transfer Learning**

The system classifies images into **34 different food categories** and displays **nutritional information** for the detected food item.
A **Flask-based web application** is developed for real-time classification, allowing users to upload images and select a model.

---

## 🎯 Features

✔️ Classifies **34 food classes**
✔️ Supports **three deep learning models**
✔️ Displays **nutritional information** for each food item
✔️ Clean **Flask REST API**
✔️ Modern **web interface** for predictions
✔️ Accurate results using **dataset balancing** and **data augmentation**
✔️ Comprehensive evaluation using **Accuracy, Precision, Recall, F1-score, ROC & AUC**

---

## 🗂️ Dataset Information

📌 **Dataset Source:** Kaggle – *Food Image Classification Dataset*
📌 **Total Classes:** 34
📌 **Total Images Selected:** 27,200
📌 **Image Size:** 256 × 256
📌 **Split:**

* Train: 80%
* Validation: 5%
* Test: 15%

Each class is also mapped to a **nutritional profile** stored in JSON format.

---

## 🧬 Food Classes (34 Categories)

`Baked Potato, Crispy Chicken, Donut, Fries, Hot Dog, Sandwich, apple_pie, Taco, Taquito, burger, butter_naan, chai, chapati, cheesecake, chicken_curry, chole_bhature, dal_makhani, dhokla, fried_rice, ice_cream, idli, jalebi, kaathi_rolls, kadai_paneer, kulfi, masala_dosa, momos, omelette, paani_puri, pakode, pav_bhaji, pizza, samosa, sushi`

---

## 🏗️ System Architecture

```
Dataset → Model Training → Saved Models (.h5) → Flask Backend → Web UI → Predictions
```

**Architecture Layers:**

| Layer             | Description                   |
| ----------------- | ----------------------------- |
| 🗂 Data Layer     | Dataset + Nutrition JSON      |
| 🧠 Model Layer    | Custom CNN + VGG16 + ResNet50 |
| 🔌 Backend Layer  | Flask API for prediction      |
| 🎨 Frontend Layer | Web UI for users              |

---

## 🧠 Deep Learning Models

### 1️⃣ Custom CNN

A lightweight CNN built from scratch with:

* Multiple Conv + MaxPool layers
* Flatten + Dense layers
* Softmax output for 34 categories

### 2️⃣ VGG16 (Transfer Learning)

* Pretrained on ImageNet
* Frozen convolutional layers
* Added custom dense layers
* High-quality feature extraction

### 3️⃣ ResNet50 (Transfer Learning)

* Deep residual blocks with skip connections
* Solves vanishing gradient problem
* Best accuracy among all models

---

## 📊 Model Comparison

| Feature            | Custom CNN | VGG16        | ResNet50         |
| ------------------ | ---------- | ------------ | ---------------- |
| Accuracy           | ⭐⭐⭐        | ⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐            |
| Training Time      | Fast       | Medium       | Slow             |
| Feature Extraction | Medium     | High         | Very High        |
| Best Use           | Learning   | Balanced use | Deployment-ready |

---

## 📈 Evaluation Metrics

The following metrics were calculated:

* ✔️ Confusion Matrix
* ✔️ Accuracy
* ✔️ Precision
* ✔️ Recall
* ✔️ F1-Score
* ✔️ ROC Curve
* ✔️ AUC Score

---

## 🔧 Technologies Used

### **Languages & Libraries**

* Python
* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* Matplotlib / Seaborn

### **Backend**

* Flask

### **Frontend**

* HTML / CSS / JavaScript

---

## 📦 Project Structure

```
📁 Food-Classification-Project
│── 📁 dataset/
│── 📁 models/
│     ├── custom_cnn.h5
│     ├── vgg16.h5
│     ├── resnet50.h5
│── 📁 static/
│── 📁 templates/
│── app.py
│── nutrition.json
│── README.md
```

---

## ▶️ How to Run the Project

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/food-classification.git
cd food-classification
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Flask app**

```bash
python app.py
```

### **4. Open in browser**

```
http://127.0.0.1:5000
```

---

## 🌐 Deployment

The project is fully deployable using:

* **Flask**
* **Render / Railway / Heroku**
* **Docker Container**

Models load dynamically based on user selection.

---

## 🥗 Sample Output

✔ Detected Food: **Pizza**
✔ Probability: **0.91**
✔ Calories: **266 kcal**
✔ Protein: **11g**

---

## 🏁 Conclusion

This project successfully demonstrates a scalable, accurate, and real-world applicable **deep learning food classification system** using multiple models. With strong performance from **ResNet50** and practical deployment through **Flask**, the system is suitable for:

* Nutrition tracking apps
* Food delivery automation
* Restaurant menu classification
* Healthcare diet analysis

---

## 📚 References

* TensorFlow Documentation
* VGG16 & ResNet Research Papers
* Kaggle Food Dataset
* Flask Framework
* Deep Learning Books (Goodfellow, Chollet, Géron)

---

If you want, I can also:

✅ Add a **project banner**
✅ Add **GIF demo** section
✅ Format README exactly for **GitHub appearance**
✅ Generate **requirements.txt**
✅ Generate **badge icons & shield icons**

Just tell me!
