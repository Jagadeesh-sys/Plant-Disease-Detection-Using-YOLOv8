from flask import Flask, request, render_template
import os
from PIL import Image
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("weights/best (2).pt")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the supplement info dataset
SUPPLEMENT_INFO_FILE = 'supplement_info.csv'
supplement_info_df = pd.read_csv(SUPPLEMENT_INFO_FILE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = Image.open(file_path)
    results = model(image)

    # Extract the highest confidence prediction
    if len(results[0].boxes) > 0:
        highest_confidence_index = results[0].boxes.conf.argmax()
        disease_name = results[0].names[results[0].boxes.cls[highest_confidence_index].item()]

        # Get supplement info based on the disease name
        supplement_info = supplement_info_df[supplement_info_df['disease_name'] == disease_name]
        if not supplement_info.empty:
            supplement_name = supplement_info['supplement name'].values[0]
            supplement_image = supplement_info['supplement image'].values[0]
            supplement_buylink = supplement_info['buy link'].values[0]
            disease_description = supplement_info['Description'].values[0]
        else:
            supplement_name = "No supplement found"
            supplement_image = ""
            supplement_buylink = ""
            disease_description = "No description available"

    else:
        disease_name = "No disease detected"
        supplement_name = ""
        supplement_image = ""
        supplement_buylink = ""
        disease_description = ""

    return render_template('index.html', disease_name=disease_name, image_path=file_path, supplement_name=supplement_name, supplement_image=supplement_image, supplement_buylink=supplement_buylink, disease_description=disease_description)

if __name__ == '__main__':
    app.run(debug=True)
