from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

MODEL = tf.keras.models.load_model('./Models/model_1/')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get('/ping')
async def ping():
    return 'Hello I am alive!'

def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)))
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    prediction = MODEL.predict(image_batch)

    class_name = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), ndigits=2)
    
    return {
        'class': class_name,
        'confidence': confidence
    }

if __name__ == 'main':
    uvicorn.run(app, host='localhost', port=8000)