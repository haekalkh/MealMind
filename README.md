# 🧠 MealMind – Personalized Food Detection for Better Nutrition

MealMind is a computer vision-based project that identifies food types from images, calculates nutritional value, and matches it to the user's health profile (BMI, age, medical history). It supports the "Makan Bergizi Gratis" initiative by ensuring each portion matches personal dietary needs.

---

## 🚀 Features

- 🍱 Food detection using YOLOv8
- 🔬 Nutrition estimation using food datasets
- 📊 Integration with BMI and daily nutrient needs
- 📱 Ready for deployment to Flutter app
- 🧠 Designed for health personalization

---

## 🗂️ Folder Structure

MealMind/
├── src/                   # Source code utama (deteksi makanan, AI, preprocessing, dsb)
│   └── main.py
├── assets/                # Gambar, sample dataset, model YOLO, dll
├── docs/                  # Dokumen pendukung (flowchart, literatur, catatan)
├── requirements.txt       # Library yang dibutuhin
├── .gitignore             # Supaya file sampah gak ke-track
├── README.md              # Penjelasan project lo
└── LICENSE                # Lisensi (MIT)

1. Clone this repository:

```bash
git clone https://github.com/haekalkh/MealMind.git
cd MealMind

2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the main script (sample):
python src/main.py
```

## 🧾 Dependencies:
opencv-python
ultralytics
numpy
pillow



