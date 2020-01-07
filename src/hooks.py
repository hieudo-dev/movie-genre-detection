import eel
import os
from keras.models import load_model


model = None

@eel.expose
def lm():
    pass
    # path = os.getcwd()
    # model = load_model(os.path.join(path, "dist", "model.h5"))

@eel.expose
def predict():
    pass
