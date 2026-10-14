from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import json

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit 16MB

# Load the models
model1 = tf.keras.models.load_model('model1/effnet.keras')
model2 = tf.keras.models.load_model('model2/vgg.h5')

# Labels
labels1 = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
labels2 = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# A function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Save predictions to a file
def save_prediction(filename, prediction, model_key):
    predictions_file = f'predictions{model_key}.json'
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}

    predictions[filename] = prediction

    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

def clear_upload_folder():
    """Remove all files from the uploads folder."""
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Clear the uploads folder before saving the new file
            clear_upload_folder()

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            image = preprocess_image(file_path)

            # Predictions for both models
            prediction1 = model1.predict(image)
            prediction2 = model2.predict(image)
            predicted_class1 = labels1[np.argmax(prediction1)]
            predicted_class2 = labels2[np.argmax(prediction2)]

            # Save predictions
            save_prediction(filename, predicted_class1, 1)
            save_prediction(filename, predicted_class2, 2)

            return redirect(url_for('index'))

    # Load the list of uploaded files
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])

    # Load predictions for both models
    predictions1 = {}
    predictions2 = {}
    if os.path.exists('predictions1.json'):
        with open('predictions1.json', 'r') as f:
            predictions1 = json.load(f)
    if os.path.exists('predictions2.json'):
        with open('predictions2.json', 'r') as f:
            predictions2 = json.load(f)

    # Combine predictions into one dictionary
    predictions = {
        filename: {
            'model1': predictions1.get(filename, 'No prediction yet'),
            'model2': predictions2.get(filename, 'No prediction yet'),
        }
        for filename in uploaded_files
    }

    return render_template('index.html', predictions=predictions, uploaded_files=uploaded_files)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
