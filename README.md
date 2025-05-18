# Pet Breed Classifier Web App

A Flask web application that uses a TensorFlow Lite model to classify pet (cat and dog) breeds from uploaded images. The app is built with Flask and styled using Tailwind CSS.

## Features

- Upload images through a modern, responsive interface
- Real-time image preview
- Fast breed classification using TensorFlow Lite
- Displays prediction results with confidence scores
- Mobile-friendly design

## Local Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   flask run
   ```
5. Open http://localhost:5000 in your browser

## Deployment

### Render Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `flask run --host=0.0.0.0 --port=$PORT`

The `render.yaml` file in this repository already contains the necessary configuration.

### Vercel Deployment

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. Deploy:
   ```bash
   vercel
   ```

The `vercel.json` file in this repository contains the necessary configuration.

## Project Structure

```
├── app.py                 # Main Flask application
├── pet_breed_classifier.tflite  # TensorFlow Lite model
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── render.yaml           # Render deployment configuration
├── vercel.json           # Vercel deployment configuration
├── static/
│   └── uploads/          # Directory for uploaded images
└── templates/
    └── index.html        # Main application template
```

## Technical Details

- The application uses TensorFlow Lite for efficient inference
- Images are preprocessed to 224x224 pixels before classification
- The model expects RGB images normalized to [0,1]
- The web interface is built with Tailwind CSS for a modern look
- File uploads are restricted to images and have a 16MB size limit

## License

MIT License 