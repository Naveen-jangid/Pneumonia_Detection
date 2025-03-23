from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.responses import HTMLResponse
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# Define image size
IMAGE_SIZE = (224, 224)

# Function to preprocess uploaded image
def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API Endpoint to make predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)[0][0]

        # Convert prediction to label
        label = "Pneumonia" if prediction > 0.5 else "Normal"

        return {"prediction": label, "confidence": float(prediction)}
    
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/", response_class=HTMLResponse)
def serve_html():
    with open("index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
