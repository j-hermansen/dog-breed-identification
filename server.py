from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import csv

print("TF version:", tf.__version__)
print("TF Hub version:", hub.__version__)

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Needed for flashing messages

# Load your model
model = tf.keras.models.load_model('models/20230724-15151690211709-full-image-set-mobilenetv2-Adam.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Get all breeds
breeds = []
with open('labels.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists. If there's no header, you can comment out this line.
    for row in reader:
        breeds.append(row[1])

unique_breeds = list(set(breeds))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Convert the file to an image
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image (based on your model's requirements)
            image = image.resize((224, 224))
            image_array = np.asarray(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Get the prediction
            predictions = model.predict(image_array)
            predicted_breed = unique_breeds[np.argmax(predictions[0])] # getting the breed with highest prediction
            print("prediction breed", predicted_breed)

            # Flash a message with the result
            flash(f'Predicted breed: {predicted_breed}')
            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
