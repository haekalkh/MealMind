# MealMind V4 🍽️🧠
> AI-Based Food Detection and Nutritional Analysis System

---

## 📜 Deskripsi

MealMind V4 adalah sebuah sistem berbasis **Artificial Intelligence (AI)** yang mampu:
- Mendeteksi berbagai jenis makanan dari gambar menggunakan model YOLOv8.
- Menghitung total nutrisi dari makanan yang terdeteksi (kalori, protein, karbohidrat, lemak, serat).
- Menghitung **BMI (Body Mass Index)** pengguna berdasarkan berat dan tinggi badan.
- Mengestimasi kebutuhan kalori harian pengguna.
- Memberikan analisis apakah piring tersebut sesuai dengan kebutuhan kalori harian.
- Menyimpan hasil analisis dalam format `.txt` dan `.json`.

---

## 🛠️ Teknologi yang Digunakan

- Python 3.x
- Ultralytics YOLOv8
- Pandas
- OpenCV
- Matplotlib

---

## 📦 Instalasi

Clone repository ini:

```bash
git clone https://github.com/username/MealMindV4.git
cd MealMindV4

## Install semua dependensi yang diperlukan:
pip install -r requirements.txt

Pastikan folder MealMindV4 memiliki:

Model YOLOv8 (model/best.pt)

Database nutrisi (data/mealmind_v4_nutrisi.csv)

Gambar makanan untuk inference (results/test_image_2.jpg)

##Jalankan program MealMind:
python mealmind_v4.py

