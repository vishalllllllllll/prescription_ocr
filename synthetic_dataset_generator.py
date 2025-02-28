import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont

# Define prescription components
patients = ["John Doe, Age: 45", "Alice Smith, Age: 34", "Robert Brown, Age: 50", "Emma Wilson, Age: 29","patrick bateman , Age: 30"]
medicines = ["Paracetamol 500mg", "Ibuprofen 200mg", "Amoxicillin 250mg", "Metformin 500mg", "salbutamol 250mg"]
dosages = ["Twice Daily", "Once Daily", "Thrice Daily", "Morning & Night",]
doctors = ["Dr. Smith", "Dr. Johnson", "Dr. Lee", "Dr. Patel", "dr.diwagar"]

# Create dataset folders
dataset_dir = "synthetic_prescription_dataset"
images_dir = os.path.join(dataset_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# CSV file for labels
csv_file_path = os.path.join(dataset_dir, "labels.csv")

# Load a handwriting-style font
font_path = "C:/Users/visha/prescription_ocr/dataset/IndieFlower-Regular.ttf"  # 🔹 Replace with a real handwriting font file path
try:
    font = ImageFont.truetype(font_path, 28)
except IOError:
    print("⚠️ Error: Font file not found! Download a handwriting font and update font_path.")
    exit()

# Generate images
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Extracted Text"])

    for i in range(500):  # Generate 500 samples
        patient = random.choice(patients)
        medicine = random.choice(medicines)
        dosage = random.choice(dosages)
        doctor = random.choice(doctors)
        
        text = f"Patient: {patient}\nMedicine: {medicine}\nDosage: {dosage}\nDoctor: {doctor}"
        img = Image.new("RGB", (500, 300), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), text, fill="black", font=font)
        
        img_name = f"prescription_{i+1}.jpg"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        
        writer.writerow([img_name, text])

print(f"✅ Synthetic dataset generated! Check the folder: {dataset_dir}")
