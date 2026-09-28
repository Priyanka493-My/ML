import numpy as np
from fastapi import FastAPI
from PIL import Image
import tensorflow as tf
from keras.applications.xception import preprocess_input
from keras_image_helper import create_preprocessor

app = FastAPI()

preprocessor = create_preprocessor('xception',target_size=(150,150))

interpreter = tf.lite.Interpreter(model_path = 'clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt',
]

def predict_url(url):

    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index,X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    predictions = preds[0].tolist()
    return dict(zip(classes,predictions))

@app.post('/predict')
async def predict_endpoint(payload:dict):
    url = payload.get('url')
    result = predict_url(url)
    return result
