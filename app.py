import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

import tensorflow as tf


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='pet_breed_classifier.tflite')

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model input details for debugging
print("Model Input Details:")
print(f"Input Shape: {input_details[0]['shape']}")
print(f"Input Type: {input_details[0]['dtype']}")

# Load breed labels (you might want to replace this with your actual labels)
BREED_LABELS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "american bulldog", "american pit bull terrier",
    "basset hound", "beagle", "boxer", "chihuahua", "english cocker spaniel",
    "english setter", "german shorthaired", "great pyrenees", "havanese",
    "japanese chin", "keeshond", "leonberger", "miniature pinscher",
    "newfoundland", "pomeranian", "pug", "saint bernard", "samoyed",
    "scottish terrier", "shiba inu", "staffordshire bull terrier",
    "wheaten terrier", "yorkshire terrier"
]

def preprocess_image(image_path):
    """Preprocess the image for model input."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Changed from 224x224 to 128x128 to match model's expected input
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array[np.newaxis, ...]

def predict_breed(image_path):
    """Run inference on the image and return the predicted breed."""
    # Preprocess the image
    input_data = preprocess_image(image_path)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class
    predicted_class = np.argmax(output_data[0])
    confidence = float(output_data[0][predicted_class])
    
    return BREED_LABELS[predicted_class], confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Get prediction
            breed, confidence = predict_breed(file_path)
            
            # Return result
            return jsonify({
                'breed': breed,
                'confidence': f"{confidence:.2%}",
                'image_path': f"/static/uploads/{filename}"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 