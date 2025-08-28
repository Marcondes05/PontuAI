import pickle

def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
