import json
from paddleocr import PaddleOCR
import re

# Initialize OCR once (to avoid reloading it every request)
ocr = PaddleOCR()

def extract_text(image_path):
    """Extracts text from an image using PaddleOCR."""
    result = ocr.ocr(image_path, cls=True)
    extracted_text = "\n".join([entry[1][0] for line in result for entry in line if entry[1]])

    # Fix common OCR misreadings
    corrections = {
        "Aqe": "Age",
        "5o": "50",
        "Dosaqe": "Dosage",
        "25omg": "250mg",
        "50omg": "500mg",
        "2oomg": "200mg",
        "20omg":"200mg"
    }
    for wrong, correct in corrections.items():
        extracted_text = extracted_text.replace(wrong, correct)

    return extracted_text

def parse_prescription(text):
    """Parses structured data (Name, Age, Medicine, Dosage, Doctor) from OCR text."""
    parsed_data = {}

    patient_pattern = re.search(r'Patient[:\s]+([\w\s]+)[,\n]?\s*Age[:\s]+(\d+)', text, re.IGNORECASE)
    medicine_pattern = re.search(r'Medicine[:\s]+([^\n]+)', text, re.IGNORECASE)
    dosage_pattern = re.search(r'Dosage[:\s]+([^\n]+)', text, re.IGNORECASE)
    doctor_pattern = re.search(r'Doctor[:\s]+([^\n]+)', text, re.IGNORECASE)

    if patient_pattern:
        parsed_data['Patient Name'] = patient_pattern.group(1).strip()
        parsed_data['Age'] = patient_pattern.group(2).strip()
    if medicine_pattern:
        parsed_data['Medicine'] = medicine_pattern.group(1).strip()
    if dosage_pattern:
        parsed_data['Dosage'] = dosage_pattern.group(1).strip()
    if doctor_pattern:
        parsed_data['Doctor'] = doctor_pattern.group(1).strip()

    return parsed_data

def run_inference(image_path):
    """Runs OCR on an image and returns structured data."""
    extracted_text = extract_text(image_path)
    structured_data = parse_prescription(extracted_text)
    return structured_data


