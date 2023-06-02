from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf

app = FastAPI()

MODEL = tf.saved_model.load("../saved_model")


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = ImageOps.fit(image, (224, 224))  # Resize and crop the image to (224, 224)
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    image_batch = tf.cast(image_batch, tf.float32)

    # call the model on the test data
    result = MODEL.signatures['serving_default'](tf.constant(image_batch))

    prediction = result['output_layer']
    prediction = prediction.numpy()[0]

    if prediction < 0.5:
        label = 'cat'
    else:
        label = 'dog'

    return {
        'result': label,
    }

    # return image

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)