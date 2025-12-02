# **🐟Deteksi Ikan Kerapu Sunu (PCD Klasik Non-ML)**

**Nama Proyek:** Deteksi Ikan Kerapu Sunu (*Plectropomus leopardus*) dengan Pengolahan Citra Digital Klasik

## **Deskripsi Singkat**

Proyek ini menghadirkan sistem deteksi objek yang **sepenuhnya berbasis Pengolahan Citra Digital (PCD) klasik**, dirancang untuk mengidentifikasi Ikan Kerapu Sunu (*Plectropomus leopardus*) di lingkungan akuatik. Sistem ini memadukan analisis **Warna, Bentuk, dan Tekstur Bintik** menggunakan metrik *scale-invariant* (persentase) untuk memastikan akurasi tanpa ketergantungan pada Machine Learning.

## **Fitur Utama**

* **UI Interaktif (PyQt5):** Antarmuka pengguna grafis yang mudah digunakan untuk memuat citra, memproses deteksi, dan menghasilkan laporan.  
* **Analisis Multi-Fitur:** Klasifikasi didasarkan pada tiga kriteria utama:  
  1. **Warna & Segmentasi:** Menggunakan ruang warna **HSV** untuk mengisolasi tubuh ikan (rentang merah/oranye).  
  2. **Bentuk:** Analisis **Circularity** dan **Aspect Ratio** untuk memverifikasi bentuk tubuh yang memanjang.  
  3. **Tekstur Bintik (Kecerahan):** Deteksi bintik dilakukan dengan mengisolasi piksel **Kecerahan Tinggi (Value HSV \> 200\)**.  
* **Deteksi Bintik Invarian Skala:** Keberadaan bintik divalidasi dengan menghitung **Total Area Kontur Bintik** dan membandingkannya dengan **persentase dari Total Area Ikan** (misalnya, minimum 1.0%), membuat sistem robust terhadap resolusi gambar.  
* **Robustness Segmentasi:**  
  * **Morfologi CLOSE:** Diterapkan pada *mask* ikan untuk menutup lubang yang disebabkan oleh bintik atau bayangan.  
  * **Masking Ganda:** Memastikan bintik hanya dihitung **di dalam** area tubuh ikan yang tersegmentasi (mengatasi *False Positive* di *background*).  
* **Laporan Otomatis:** Menghasilkan file **HTML Laporan** yang visual dan terperinci untuk mendokumentasikan setiap langkah pemrosesan.

---

## **💻 Kebutuhan Sistem (Dependencies)**

Proyek ini memerlukan Python 3.x dan pustaka berikut:

Bash

```pip install opencv-python numpy PyQt5```

---

## **⚙️ Panduan Penggunaan**

### **1\. Struktur Proyek**

deteksi\_kerapu\_sunu/  
├── sunu\_finder.py      \# Logika PCD dan UI PyQt5  
└── README.md           \# Dokumen ini

### **2\. Menjalankan Aplikasi**

1. Buka terminal di direktori proyek.  
2. Jalankan file Python utama:  
   Bash  
   python sunu\_finder.py

3. **Alur Kerja di Aplikasi:**  
   * Klik 1\. Input Gambar.  
   * Klik 2\. Proses Deteksi (Menjalankan semua 6 langkah PCD).  
   * Klik 3\. Generate Laporan (HTML) untuk menyimpan laporan teknis.

### **3\. Alur Pemrosesan Visual**

Sistem memproses citra melalui tahapan yang divisualisasikan:

| Langkah | Deskripsi Proses | Analisis Kunci |
| :---- | :---- | :---- |
| **A/B** | Input & Prapemrosesan Grayscale | Menghilangkan *noise*. |
| **C/D** | Segmentasi Warna & Masking | Isolasi Ikan (Mask Biner ditutup lubangnya). |
| **E** | Deteksi Bintik (Visual) | **Analisis Kontur Kecerahan Tinggi** (Visualisasi *bounding box* bintik yang dihitung). |
| **F** | Klasifikasi Akhir | **Validasi Bentuk** dan **Persentase Area Bintik**. |
