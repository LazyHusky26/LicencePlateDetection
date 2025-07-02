# 🚗 Number Plate Detection from Video

A web application for detecting vehicle number plates from uploaded video files. The app uses a YOLO model for number plate detection and Tesseract OCR for reading the plate text. Results are visualized and can be downloaded as an Excel sheet.

## ✨ Features

- Detects number plates from videos using a custom-trained YOLO model.
- Applies OCR on detected plates to extract readable text.
- Uses consensus logic for more accurate text recognition across frames.
- Search for specific number plates in detection results.
- View the exact frame and bounding box for any detected plate.
- Download detection results as an Excel file.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Detection Model**: YOLO (Ultralytics)
- **OCR**: Tesseract
- **Data Export**: pandas (Excel via `to_excel`)

---

