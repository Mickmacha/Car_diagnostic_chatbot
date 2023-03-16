import pickle
from . import main
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)
if __name__ == "__main__":
    main.chat()