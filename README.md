# 🏥 Prescription OCR - Handwritten Text Recognition

## 📌 Overview
This project is an AI-powered **handwritten prescription OCR model** that extracts and structures medical data from handwritten doctor prescriptions. The extracted text is formatted into a structured table format, making it easier to read and analyze.

## 🚀 Features
✅ Extracts key details like **Patient Name, Age, Medicines, Dosage, and Doctor Name**  
✅ Uses **PaddleOCR** for text recognition  
✅ Provides a **web-based interface** for easy image upload and result display  
✅ **Flask backend** for running inference and processing requests  
✅ Displays extracted data in a **clean and structured table format**  

---
## 🔧 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/prescription_ocr.git
cd prescription_ocr
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Flask Server**
```sh
python app.py
```

The server will start at `http://127.0.0.1:5000/`

### **4️⃣ Open the Web Interface**
- Open `index.html` in a browser
- Upload a prescription image
- Get the extracted data in a structured table format

---
## 📸 Example Output
**Input Image:** Handwritten doctor prescription  
**Extracted Data (JSON Format):**
```json
{
    "table": [
        { "Field": "Patient Name", "Value": "Robert Brown" },
        { "Field": "Age", "Value": "50" },
        { "Field": "Medicine", "Value": "salbutamol 250mg" },
        { "Field": "Dosage", "Value": "Thrice Daily" },
        { "Field": "Doctor", "Value": "Dr. Smith" }
    ]
}
```

---
## 🎯 Deployment (GitHub Pages & Flask API)
### **1️⃣ Deploy Frontend using GitHub Pages**
1. Push `index.html` and other frontend files to GitHub
2. Go to **Settings → Pages → Select `main` branch**
3. Access your website at:
   ```
   https://vishalllllllllll.github.io/prescription_ocr/
   ```

### **2️⃣ Deploy Backend using Render/Heroku**
- Use **Render** or **Heroku** to deploy the Flask API
- Update `fetch()` URL in `index.html` to the deployed API endpoint

---
## 💡 Future Improvements
🔹 Improve OCR accuracy using **custom-trained deep learning models**  
🔹 Support for **multiple languages and prescription formats**  
🔹 Secure API with authentication and **database integration**  

---
## 👨‍💻 Author
**Vishal**  
📧 Contact: [vishalmuru6@gmail.com]  
🔗 GitHub: https://github.com/vishalllllllllll

---
## 📝 License
This project is open-source and available under the **MIT License**.
