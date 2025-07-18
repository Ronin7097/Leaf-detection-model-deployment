# Leaf Prediction Application

This project is a web application designed to predict leaf diseases from uploaded images or live camera feeds. It consists of a Flask backend that handles image processing and machine learning predictions, and a React frontend for a user-friendly interface.

## Features

- Upload leaf images for disease prediction.
- Use your webcam to capture and predict leaf on the spot.
- Displays predicted plant class and confidence level.

Follow these steps to set up and run the project locally.

### 1. Backend Setup (Flask)

Navigate to the `client` directory and install the required Python packages.

```bash
cd client
pip install -r requirements.txt
```

**Note:** The `densenet169_best_final.pth` model file is required for the backend to function. Ensure this file is present in the `client/` directory.

### 2. Frontend Setup (React)

Navigate to the `Leaf_Detection` directory and install the required Node.js packages.

```bash
cd Leaf_Detection
npm install
```

## Running the Application

### 1. Start the Backend Server

Open a new terminal, navigate to the `client` directory, and run the Flask application:

```bash
cd client
python app.py
```
### 2. Start the Frontend Development Server

Open another new terminal, navigate to the `Leaf_Detection` directory, and start the React development server:

```bash
cd Leaf_Detection
npm run dev
```
