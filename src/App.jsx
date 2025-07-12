import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Webcam from "react-webcam";
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState('');
  const [preview, setPreview] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const webcamRef = useRef(null);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction('');
      setConfidence('');
      setCameraActive(false); // Close camera if a file is selected
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file or capture an image first!');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error('Error uploading the file:', error);
      setPrediction('Error making prediction.');
      setConfidence('');
    }
  };

  const capturePhoto = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        fetch(imageSrc)
          .then(res => res.blob())
          .then(blob => {
            const capturedFile = new File([blob], "captured_image.jpeg", { type: "image/jpeg" });
            setSelectedFile(capturedFile);
            setPreview(URL.createObjectURL(capturedFile));
            setCameraActive(false); // Close camera after capturing
          });
      }
    }
  };

  return (
    <div className="container">
      <h1>Leaf Disease Prediction</h1>
      <div className="upload-section">
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button onClick={handleUpload}>Predict</button>
      </div>

      <div className="camera-section">
        {!cameraActive && <button onClick={() => setCameraActive(true)}>Open Camera</button>}
        {cameraActive && (
          <>
            <Webcam
              audio={false}
              height={360}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={640}
              videoConstraints={videoConstraints}
            />
            <button onClick={capturePhoto}>Capture Photo</button>
            <button onClick={() => setCameraActive(false)}>Close Camera</button>
          </>
        )}
      </div>

      {preview && (
        <div className="image-preview">
          <img src={preview} alt="Selected leaf" />
        </div>
      )}
      {prediction && (
        <div className="prediction">
          <h2>Prediction:</h2>
          <p>{prediction}</p>
          {confidence && <p>Confidence: {confidence}</p>}
        </div>
      )}
    </div>
  );
}

export default App;