## **🐠 Analisis Proyek Deteksi Ikan Kerapu Sunu (PCD Klasik)**

Proyek ini menantang karena harus mengandalkan teknik PCD klasik (seperti berbasis fitur, warna, bentuk, atau tekstur) alih-alih pembelajaran mendalam (Deep Learning).

---

## **1\. Karakteristik Ikan Kerapu Sunu sebagai Fitur Kunci**

Langkah pertama yang krusial adalah mengidentifikasi **fitur unik dan stabil** dari Ikan Kerapu Sunu (*Plectropomus leopardus*) yang dapat diekstraksi secara klasik.

* **Fitur Warna dan Bintik (Utama):**  
  * **Warna Dasar:** Abu-abu kehijauan hingga merah cerah (tergantung variasi dan kondisi).  
  * **Pola Khas:** Seluruh kepala, badan, dan sirip ditutupi oleh **bintik-bintik berwarna biru terang dengan tepi gelap**. Bintik-bintik ini umumnya **berukuran kecil dan seragam**.  
  * **Bintik Khusus:** Ada satu bintik biru gelap yang khas di **pangkal sirip dada**.  
* **Fitur Bentuk (Sekunder):**  
  * **Bentuk Tubuh:** Agak **gepeng dan memanjang (memanjang tegak)**.  
  * **Sirip Ekor:** Ujung sirip ekor biasanya **rata** dengan garis putih di ujungnya.  
* **Keterbatasan:** Warna ikan kerapu dapat berubah dipengaruhi oleh kondisi lingkungan dan tingkat stres, yang dapat menjadi tantangan utama dalam deteksi berbasis warna murni.

---

## **2\. Pilihan Metode PCD Klasik**

Berdasarkan fitur-fitur di atas, metode PCD klasik yang dapat dipertimbangkan meliputi:

### **A. Segmentasi Berbasis Warna (Color-Based Segmentation)**

* **Tujuan:** Memisahkan ikan (objek) dari latar belakang (karang, air).  
* **Langkah-langkah:**  
  1. **Konversi Ruang Warna:** Mengubah citra dari RGB ke ruang warna yang lebih robust terhadap perubahan iluminasi, seperti **HSV** atau **$L\\cdot a\\cdot b$** (di mana komponen *H* atau *a/b* lebih sensitif terhadap warna target—merah/biru—daripada *V* atau *L*).  
  2. **Thresholding:** Menentukan rentang (*range*) nilai warna (misalnya, rentang HSV untuk warna merah/kehijauan pada ikan) untuk membuat citra biner (mask).  
  3. **Morfologi:** Menerapkan operasi **Erosi** dan **Dilasi** untuk menghilangkan *noise* dan merapikan tepi objek.

### **B. Ekstraksi Fitur Tekstur dan Bentuk (Feature Extraction)**

* **Tujuan:** Mengidentifikasi pola bintik-bintik dan bentuk ikan.  
* **Metode Tekstur (Untuk Bintik):**  
  * **Gray-Level Co-occurrence Matrix (GLCM):** Dapat digunakan untuk mengekstrak fitur tekstur (seperti *Contrast*, *Energy*, atau *Homogeneity*) dari pola bintik biru yang seragam.  
  * **Filter Gabor:** Efektif untuk mendeteksi pola dan frekuensi tertentu (yaitu, pola bintik-bintik yang berulang).  
* **Metode Bentuk (Untuk Outline Ikan):**  
  * **Chain Code atau Contour Tracing:** Setelah segmentasi, digunakan untuk melacak batas luar (kontur) objek.  
  * **Shape Descriptors (misalnya, Moments Invariant):** Digunakan untuk mengukur bentuk objek (seperti luasan, perimeter, rasio aspek, *solidity*) yang dapat membedakan bentuk Kerapu Sunu dari objek laut lainnya (misalnya, batu atau karang).

### **C. Pendeteksian Objek Menggunakan Feature Descriptors (Deteksi Berbasis Titik Kunci)**

* **Metode:** **SIFT** (*Scale-Invariant Feature Transform*) atau **SURF** (*Speeded-Up Robust Features*).  
* **Tujuan:** Mengidentifikasi titik-titik unik (keypoints) yang stabil pada pola bintik biru atau bentuk tubuh ikan, dan mencocokkannya dengan *keypoints* dari citra referensi Kerapu Sunu.  
* **Proses:**  
  1. Ambil citra Kerapu Sunu sebagai **Template**.  
  2. Ekstrak *keypoints* (misalnya, bintik-bintik) dari template dan citra uji.  
  3. Lakukan pencocokan *keypoints* (misalnya, menggunakan *Brute-Force Matcher*).  
  4. Hitung transformasi homografi untuk memverifikasi lokasi objek yang terdeteksi.

---

## **3\. Tahapan Implementasi Proyek (Rancangan Sistem)**

Proyek dapat dibagi menjadi tiga tahap utama:

### **1\. Tahap Pra-pemrosesan (Preprocessing) 🛠️**

| Langkah | Deskripsi | Tujuan |
| :---- | :---- | :---- |
| **Akuisisi Citra** | Kumpulkan dataset citra/video Kerapu Sunu (di bawah air atau di bak/tangki) dengan berbagai kondisi pencahayaan dan latar belakang. | Data masukan. |
| **Peningkatan Citra** | Terapkan filter **Median** (untuk menghilangkan *salt-and-pepper noise* bawah air) dan **penyesuaian kontras** (*Histogram Equalization*). | Memperjelas fitur ikan dari latar belakang. |

### **2\. Tahap Pemrosesan (Processing) ⚙️**

| Strategi | Deskripsi | Hasil |
| :---- | :---- | :---- |
| **Segmentasi Awal** | Konversi ke ruang warna HSV atau $L\\cdot a\\cdot b$. Terapkan **thresholding** untuk mengisolasi area berwarna merah/biru bintik ikan. | Citra Biner (Mask) yang mengisolasi objek ikan. |
| **Ekstraksi Fitur** | Terapkan algoritma **Connected Component Labeling** pada mask untuk mengidentifikasi objek-objek terpisah. **Hitung fitur bentuk** (*area, perimeter, circularity, moments invariant*) dan **fitur tekstur** (GLCM) untuk setiap objek. | Vektor Fitur (bentuk dan tekstur) untuk setiap kandidat objek. |

### **3\. Tahap Klasifikasi/Deteksi (Classification/Detection) ✅**

| Langkah | Deskripsi | Hasil |
| :---- | :---- | :---- |
| **Klasifikasi Berbasis Aturan** | Terapkan serangkaian **aturan (*if-then*)** berdasarkan rentang nilai fitur yang telah ditentukan (misalnya, *Jika Area \> MinArea* **AND** *Solidity \> MinSolidity* **AND** *Contrast\_GLCM* berada di rentang yang menunjukkan bintik-bintik\*). | Keputusan: Objek adalah Ikan Kerapu Sunu atau Bukan. |
| **Visualisasi Hasil** | Gambarkan kotak pembatas (*bounding box*) di sekitar objek yang terdeteksi sebagai Kerapu Sunu pada citra asli. | Output akhir: Citra dengan deteksi objek. |

---

## **4\. Tantangan dan Mitigasi**

| Tantangan | Deskripsi | Mitigasi dalam PCD Klasik |
| :---- | :---- | :---- |
| **Variasi Iluminasi** | Cahaya bawah air yang bervariasi atau bayangan dapat mengubah nilai RGB/HSV. | Gunakan ruang warna yang lebih stabil ($L\\cdot a\\cdot b$ atau HSV) dan terapkan teknik peningkatan kontras. |
| **Latar Belakang Kompleks** | Karang dengan pola dan warna yang mirip dengan ikan (misalnya, karang berbintik). | Andalkan **kombinasi fitur**: Bentuk (memanjang) \+ Bintik (GLCM) \+ Bintik Punggung (Template Matching untuk bintik spesifik). |
| **Perubahan Warna Ikan** | Warna ikan dapat memudar karena stres atau kondisi, mempengaruhi segmentasi. | **Fokuskan pada pola bintik** dan **bentuk** daripada warna dasar murni. Gunakan fitur tekstur (GLCM) dari bintik sebagai fitur utama. |