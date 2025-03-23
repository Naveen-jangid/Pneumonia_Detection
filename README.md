# Pneumonia Detection using Deep Learning

##  Project Overview
This project aims to detect **Pneumonia** from chest X-ray images using a **Deep Learning** model. The model is deployed using **FastAPI** to provide an easy-to-use web interface where users can upload an image and receive predictions.

##  Features
- Upload a chest X-ray image
- Get instant predictions (Pneumonia / Normal)
- Displays uploaded image and results on the UI
- Simple & interactive frontend

##  Tech Stack
- **Backend**: FastAPI, TensorFlow
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Pretrained CNN (Trained on Chest X-ray dataset)

##  Installation & Setup

### 1️ Clone the Repository
```sh
git clone https://github.com/Naveen-jangid/Pneumonia_Detection.git
cd Pneumonia_Detection
```

### 2️ Create & Activate Virtual Environment
```sh
python -m venv xray_venv  # Create virtual environment
source xray_venv/bin/activate  # Mac/Linux
xray_venv\Scripts\activate  # Windows
```

### 3️ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️ Run the FastAPI App
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
App will be available at: **http://127.0.0.1:8000/**

---
## Usage Guide
1. Open the web app in your browser.
2. Upload a chest X-ray image.
3. Click **Predict**.
4. View the prediction result (Pneumonia or Normal).

![image](https://github.com/user-attachments/assets/05fdbed0-9602-4090-b03c-5262c6cff581)

## Troubleshooting
- **Model Not Loading?** Ensure the model file is in the correct path.
- **Server Not Running?** Check if FastAPI and TensorFlow are installed.
- **Port Conflict?** Change the port in `main.py` (default: `8000`).

## License
This project is open-source and available under the **MIT License**.
