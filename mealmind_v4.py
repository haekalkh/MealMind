# --- Import Libraries ---
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# --- Load Model YOLO ---
model = YOLO('model/best.pt')

# --- Load Database Nutrisi ---
data_nutrisi = pd.read_csv('data/mealmind_v4_nutrisi - Sheet1.csv')
data_nutrisi.columns = data_nutrisi.columns.str.strip()  # Hapus spasi aneh
data_nutrisi = data_nutrisi.rename(columns={
    'Energi (kcal)': 'Kalori',
    'Protein (g)': 'Protein',
    'Karbohidrat (g)': 'Karbohidrat',
    'Lemak (g)': 'Lemak',
    'Serat (g)': 'Serat'
})

# --- Fungsi Helper ---
def bersihkan_nama(nama):
    return nama.replace('_', ' ').lower()

def cari_nutrisi(nama_makanan, data_nutrisi):
    nama_makanan = nama_makanan.lower()
    hasil = data_nutrisi[data_nutrisi['Nama Makanan'].str.lower() == nama_makanan]

    if not hasil.empty:
        nutrisi = hasil.iloc[0]
        return {
            'Kalori': nutrisi['Kalori'],
            'Protein': nutrisi['Protein'],
            'Karbohidrat': nutrisi['Karbohidrat'],
            'Lemak': nutrisi['Lemak'],
            'Serat': nutrisi['Serat']
        }
    else:
        return None

def hitung_bmi(berat, tinggi_cm):
    tinggi_m = tinggi_cm / 100
    bmi = berat / (tinggi_m ** 2)
    if bmi < 18.5:
        kategori = "Kurus"
    elif 18.5 <= bmi < 24.9:
        kategori = "Normal"
    elif 25 <= bmi < 29.9:
        kategori = "Overweight"
    else:
        kategori = "Obesitas"
    return bmi, kategori

def estimasi_kalori_harian(berat, kategori):
    if kategori == "Kurus":
        return berat * 35
    elif kategori == "Normal":
        return berat * 30
    elif kategori == "Overweight":
        return berat * 25
    else:  # Obesitas
        return berat * 20

# --- Input User Berat Badan dan Tinggi Badan ---
berat = float(input("Masukkan berat badan Anda (kg): "))
tinggi = float(input("Masukkan tinggi badan Anda (cm): "))

# --- Hitung BMI dan Kebutuhan Kalori Harian ---
bmi, kategori_bmi = hitung_bmi(berat, tinggi)
kebutuhan_kalori = estimasi_kalori_harian(berat, kategori_bmi)

print("\n=== Informasi BMI ===")
print(f"BMI Anda     : {bmi:.2f}")
print(f"Kategori BMI : {kategori_bmi}")
print(f"Estimasi kebutuhan kalori harian: {kebutuhan_kalori:.0f} kcal")

# --- Inference Gambar dengan YOLO ---
results = model.predict(source='/content/drive/MyDrive/MealMind_New/test_image_2.jpg', save=True, conf=0.25)

# --- Fix path hasil inference untuk ditampilkan ---
pred_path = os.path.join(results[0].save_dir, os.path.basename(results[0].path))

# --- Tampilkan Hasil Gambar Inference ---
img = cv2.imread(pred_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.title('Hasil Inference MealMind V4')
plt.show()

# --- Ekstrak Nama Makanan dari Hasil Deteksi ---
detected_classes = []

for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        detected_classes.append(class_name)

print("\n=== Makanan Terdeteksi ===")
print(detected_classes)

# --- Hitung Total Nutrisi ---
total_nutrisi = {
    'Kalori': 0,
    'Protein': 0,
    'Karbohidrat': 0,
    'Lemak': 0,
    'Serat': 0
}

print("\n=== Detail Nutrisi Setiap Makanan ===")
for makanan in detected_classes:
    nama_bersih = bersihkan_nama(makanan)
    nutrisi = cari_nutrisi(nama_bersih, data_nutrisi)

    if nutrisi:
        print(f"{nama_bersih.title()} ➔ {nutrisi}")
        total_nutrisi['Kalori'] += float(nutrisi['Kalori'])
        total_nutrisi['Protein'] += float(nutrisi['Protein'])
        total_nutrisi['Karbohidrat'] += float(nutrisi['Karbohidrat'])
        total_nutrisi['Lemak'] += float(nutrisi['Lemak'])
        total_nutrisi['Serat'] += float(nutrisi['Serat'])
    else:
        print(f"{nama_bersih.title()} tidak ditemukan di database.")

# --- Tampilkan Total Nutrisi Satu Piring ---
print("\n=== TOTAL NUTRISI 1 PIRING ===")
print(f"Kalori     : {total_nutrisi['Kalori']:.2f} kcal")
print(f"Protein    : {total_nutrisi['Protein']:.2f} g")
print(f"Karbohidrat: {total_nutrisi['Karbohidrat']:.2f} g")
print(f"Lemak      : {total_nutrisi['Lemak']:.2f} g")
print(f"Serat      : {total_nutrisi['Serat']:.2f} g")

# --- Analisa apakah kalori piring sesuai kebutuhan harian ---
print("\n=== Analisa Nutrisi ===")
porsi_kalori = total_nutrisi['Kalori'] / kebutuhan_kalori * 100
print(f"Porsi kalori piring ini setara dengan {porsi_kalori:.1f}% dari kebutuhan harian Anda.")

if porsi_kalori < 25:
    print("⚡ Piring ini kalorinya rendah, cocok untuk camilan atau sarapan ringan.")
elif 25 <= porsi_kalori <= 40:
    print("✅ Piring ini cukup ideal untuk sekali makan (makan siang/makan malam).")
else:
    print("⚠️ Piring ini kalorinya cukup tinggi, perhatikan porsinya ya!")

# --- Save Report ke TXT dan JSON ---

import json

# Data yang mau disave
report = {
    'BMI': round(bmi, 2),
    'Kategori BMI': kategori_bmi,
    'Kebutuhan Kalori Harian': round(kebutuhan_kalori),
    'Makanan Terdeteksi': detected_classes,
    'Total Nutrisi': {
        'Kalori': round(total_nutrisi['Kalori'], 2),
        'Protein': round(total_nutrisi['Protein'], 2),
        'Karbohidrat': round(total_nutrisi['Karbohidrat'], 2),
        'Lemak': round(total_nutrisi['Lemak'], 2),
        'Serat': round(total_nutrisi['Serat'], 2)
    },
    'Persentase Kebutuhan Harian': f"{porsi_kalori:.1f}%"
}

# --- Save ke TXT ---
txt_report = f"""
=== MealMind V4 Report ===

BMI Anda: {report['BMI']}
Kategori BMI: {report['Kategori BMI']}
Estimasi Kebutuhan Kalori Harian: {report['Kebutuhan Kalori Harian']} kcal

Makanan Terdeteksi:
{', '.join(report['Makanan Terdeteksi'])}

Total Nutrisi 1 Piring:
- Kalori     : {report['Total Nutrisi']['Kalori']} kcal
- Protein    : {report['Total Nutrisi']['Protein']} g
- Karbohidrat: {report['Total Nutrisi']['Karbohidrat']} g
- Lemak      : {report['Total Nutrisi']['Lemak']} g
- Serat      : {report['Total Nutrisi']['Serat']} g

Porsi kalori piring ini setara dengan {report['Persentase Kebutuhan Harian']} dari kebutuhan harian Anda.
"""

# Nama file
filename_txt = '/content/drive/MyDrive/MealMind_New/meal_report.txt'
filename_json = '/content/drive/MyDrive/MealMind_New/meal_report.json'

# Simpan
with open(filename_txt, 'w') as f:
    f.write(txt_report)

with open(filename_json, 'w') as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Report berhasil disimpan ke:")
print(f"- {filename_txt}")
print(f"- {filename_json}")