from flask_cors import CORS
from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from inference import run_inference

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allows all origins for /predict

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"  # Allowed methods
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"  # Allowed headers
    return response

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":  # Handle preflight request
        return jsonify({"message": "CORS preflight check"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    # Convert image bytes to OpenCV format
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save temp image for OCR processing
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image)

    print("✔ Image received & saved as temp.jpg")  # Debugging

    # Run inference
    structured_data = run_inference(temp_path)

    print("✔ Extracted Data:", structured_data)  # Debugging

    # Convert extracted data into table format
    table_data = [
        {"Field": "Patient Name", "Value": structured_data.get("Patient Name", "N/A")},
        {"Field": "Age", "Value": structured_data.get("Age", "N/A")},
        {"Field": "Medicine", "Value": structured_data.get("Medicine", "N/A")},
        {"Field": "Dosage", "Value": structured_data.get("Dosage", "N/A")},
        {"Field": "Doctor", "Value": structured_data.get("Doctor", "N/A")}
    ]

    # Remove temp image
    os.remove(temp_path)

    return jsonify({"table": table_data})

if __name__ == "__main__":
    app.run(debug=True)


