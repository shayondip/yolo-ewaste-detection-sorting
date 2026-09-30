# ♻️ YOLO E-Waste Detection & Sorting

A computer vision-based **E-Waste Detection and Sorting System** using **YOLO (You Only Look Once)** for detecting and classifying electronic waste objects from images.

The system is designed to identify different categories of e-waste and can be extended toward automated waste segregation and intelligent recycling systems.

---

## 🚀 Project Overview

Electronic waste contains valuable recyclable materials as well as potentially hazardous components. Manual identification and sorting of e-waste can be time-consuming and difficult to scale.

This project uses a **YOLO-based object detection pipeline** to automatically detect e-waste items and classify them into predefined categories.

### 🎯 Main Objectives

* Detect electronic waste objects automatically.
* Classify detected objects into different e-waste categories.
* Provide bounding boxes and confidence scores for detected objects.
* Build a lightweight computer-vision pipeline suitable for real-world applications.
* Create a foundation for automated e-waste sorting systems.

---

## 🧠 E-Waste Categories

The system can be configured to detect categories such as:

| Category        | Examples                       |
| --------------- | ------------------------------ |
| 🔋 Battery      | Batteries and battery cells    |
| 📱 Mobile Phone | Smartphones and mobile devices |
| 🔌 Cable        | Charging/data cables and wires |
| 🟩 PCB          | Printed Circuit Boards         |
| 🧴 Plastic      | Plastic electronic components  |
| 🔩 Metal        | Metal electronic parts         |

> The exact classes depend on the dataset and training configuration used for the model.

---

## 🏗️ System Pipeline

```text
Input Image
     ↓
Image Preprocessing
     ↓
YOLO Object Detection Model
     ↓
Object Detection
     ↓
Bounding Boxes + Confidence Scores
     ↓
E-Waste Category Classification
     ↓
Sorting / Category Assignment
```

---

## 🔥 Key Features

* ⚡ Real-time object detection
* 🎯 Bounding-box based localization
* 🧠 YOLO-based deep learning model
* ♻️ E-waste category identification
* 📊 Confidence score visualization
* 🖼️ Image-based inference
* 🔄 Extensible toward automated sorting
* 💻 Easy local deployment

---

## 🛠️ Technologies Used

* **Python**
* **YOLO**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Computer Vision**
* **Deep Learning**

---

## 📂 Project Structure

```text
yolo-ewaste-detection-sorting/
│
├── datasets/
│   └── ...
│
├── runs/
│   └── ...
│
├── weights/
│   └── best.pt
│
├── detect.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the folder/file names above if your actual project structure is different.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/shayondip/yolo-ewaste-detection-sorting.git
```

### 2. Navigate to the project

```bash
cd yolo-ewaste-detection-sorting
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Model

After installing the required dependencies and placing the trained model weights in the appropriate directory:

```bash
python detect.py
```

You can modify the inference configuration according to your dataset and model.

---

## 🏋️ Model Training

The model can be trained using the Ultralytics YOLO framework.

Example:

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
```

After training, the best-performing weights can be used for inference.

---

## 📊 Model Evaluation

The trained model can be evaluated using standard object-detection metrics:

* **Precision**
* **Recall**
* **mAP@50**
* **mAP@50–95**
* **Inference Speed**

Example evaluation command:

```bash
yolo detect val model=best.pt data=data.yaml
```

---

## 🖼️ Detection Example

The model produces predictions containing:

```text
Detected Object
      ↓
Class Label
      ↓
Bounding Box
      ↓
Confidence Score
```

Example:

```text
Battery       0.94
Mobile Phone  0.91
PCB           0.87
Cable         0.83
```

---

## 🌍 Real-World Applications

This system can potentially be integrated into:

* ♻️ Automated e-waste sorting systems
* 🏭 Recycling facilities
* 📦 Smart waste-management systems
* 🤖 Robotic waste segregation
* 🏙️ Smart-city waste management
* 📱 Mobile-based waste identification
* 🔬 E-waste monitoring and analysis

---

## 🔮 Future Work

Future improvements may include:

* Real-time webcam detection
* Automated robotic sorting
* Conveyor-belt based detection
* Edge-device deployment
* Mobile application integration
* Multi-camera detection
* More diverse e-waste categories
* Model optimization for low-power devices
* Deployment using NVIDIA Jetson / Raspberry Pi
* Cloud-based monitoring dashboard

---

## 👨‍💻 Author

**Shayondip Mitra**

B.Tech — Electronics & Computer Science

GitHub:
https://github.com/shayondip

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and research purposes.
