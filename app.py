from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load your pre-trained deep learning model
model = load_model('model.h5')

# Model input size (adjust according to your model)
input_size = (112, 112)

@app.route('/')
def home():
    return render_template('index.html')

# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['file']

        # # Save the uploaded image
        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        # image_file.save(image_path)
        # Read the image using PIL
        image_for_testing = Image.open(image_file)

        # Preprocess the image for the model
        test_image = image_for_testing.resize(input_size)
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Make predictions
        y_pred = model.predict(test_image)

        return render_template('index.html', prediction=y_pred)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)


