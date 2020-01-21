import eel
import os
import base64
import io
import re
import numpy as np
from PIL import Image

from keras.models import load_model


model = None

@eel.expose
def lm():
    global model
    path = os.getcwd()
    model = load_model(os.path.join(path, "model", "model.h5"))

@eel.expose
def predict(image_base64):
    global model
    base64_data = re.sub('^data:image/.+;base64,', '', image_base64)
    imgdata = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(imgdata))
    image = image.resize((400, 400))
    matrix = np.asarray(image)

    matrix = matrix.reshape(1, 400, 400, 3)
    print(matrix.shape)
    proba = model.predict(matrix)
    probabilities = model.predict(matrix)

    top_3 = np.argsort(probabilities[0])[:-4:-1]

    print(top_3)

    path = os.getcwd()
    with open(os.path.join(path, "src", "Movies_Poster", "Multi_Label_dataset", "train.csv")) as fd:
        columns = fd.readline()[:-1]
        print(columns)
        columns = columns.split(',')[2:]

    print(columns)
    print(len(columns))
    three = columns[top_3[0]], columns[top_3[1]], columns[top_3[2]]

    return three
