from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import uvicorn
import requests


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TECHNIQUE = ['Tank', 'Armoured Vehicle', 'Vehicle', 'Artillery System', 'UAV', 'None']

model = None


def convert_to_array(file):
    """
    Converts input bytes of an image to array and resizes it
    """
    image = Image.open(BytesIO(file)).convert("RGB")
    image = np.array(image.resize((128, 128)))

    return image

@app.get("/home")
async def home():
    return "CNN model"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = convert_to_array(await file.read())
    image_batch = np.expand_dims(image, 0).tolist()

    prediction = predict_img(image_batch)

    return prediction

def predict_img(img):
    global model

    if not model:
        path = 'opt/app/models/model-v5'
        model = tf.saved_model.load(path).signatures["serving_default"]

    softmax_out = model(tf.constant(img, dtype=tf.float32))['dense_9'][0]
    predicted_technique = TECHNIQUE[np.argmax(softmax_out)]
    proba = round(float(np.max(softmax_out)), 4)

    return {
        "technique": predicted_technique,
        "probability": proba
        }


if __name__ == '__main__':
    host = '0.0.0.0'
    uvicorn.run(app, host=host, port=8000)
