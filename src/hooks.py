import eel

@eel.expose
def load_model():
    pass

# Load Model
def loadModel():
    path = os.getcwd()

    model = load_model(os.path.join(path, "dist", "model.h5"))

    return model
