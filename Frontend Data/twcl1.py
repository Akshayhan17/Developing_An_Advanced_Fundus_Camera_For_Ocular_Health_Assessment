import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from flask import Flask, request, render_template

# Load the pre-trained model
model = load_model("vgg19.h5")

app = Flask(__name__)

# Define the model_predict function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    predicted_class = np.argmax(preds, axis=1)

    class_labels = {
        0: "Normal",
        1: "Cataract",
        2: "Glaucoma",
        3: "Myopia"
    }

    prediction = class_labels.get(predicted_class[0], "Unknown")
    return prediction

@app.route('/')
def index():
    return render_template('index.html', pred="")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'User_Images', f.filename)
            f.save(filepath)

            prediction = model_predict(filepath, model)

            return render_template('prediction.html', prediction=prediction, fname=filepath)
    
    return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
