from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('tomato_leaf_disease_model.h5')

# Define the class names
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
               'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 
               'Tomato_healthy']

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file:
            # Save the uploaded file temporarily
            upload_path = os.path.join('static/uploads', file.filename)
            os.makedirs('static/uploads', exist_ok=True)
            file.save(upload_path)

            # Make prediction
            prediction, confidence = predict_image(upload_path)
            
            return render_template('index.html', 
                                prediction=prediction, 
                                confidence=f"{confidence*100:.2f}%",
                                image_path=f"uploads/{file.filename}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
