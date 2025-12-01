import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QScrollArea, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class KerapuSunuDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Ikan Kerapu Sunu (PCD Klasik)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.original_image = None
        self.image_path = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # --- Kolom Kontrol dan Hasil ---
        control_layout = QVBoxLayout()
        
        # Tombol Input Gambar
        self.btn_input = QPushButton("1. Input Gambar")
        self.btn_input.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_input)
        
        # Tombol Proses
        self.btn_process = QPushButton("2. Proses Deteksi")
        self.btn_process.clicked.connect(self.process_detection)
        self.btn_process.setEnabled(False) # Nonaktifkan sampai gambar dimuat
        control_layout.addWidget(self.btn_process)
        
        control_layout.addStretch(1)
        
        # Hasil Analisis Teks
        self.result_label = QLabel("Hasil Analisis:")
        self.result_text = QLabel("...")
        self.result_text.setStyleSheet("font-size: 14pt; font-weight: bold; color: navy;")
        
        control_layout.addWidget(self.result_label)
        control_layout.addWidget(self.result_text)
        
        control_layout.addStretch(3)
        main_layout.addLayout(control_layout, 1)

        # --- Kolom Tampilan Proses (Image Viewer) ---
        
        self.image_widgets = {}
        image_titles = ["A. Input Citra", "B. Pre-processing (Gray)", 
                        "C. Segmentasi Warna (Mask HSV)", "D. Hasil Deteksi"]
        
        image_display_layout = QVBoxLayout()
        
        # Gunakan QScrollArea agar tampilan tidak melebihi batas layar
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        self.image_grid_layout = QHBoxLayout(scroll_content)
        
        for title in image_titles:
            step_layout = QVBoxLayout()
            title_label = QLabel(f"### {title}")
            image_label = QLabel("Tidak Ada Gambar")
            image_label.setFixedSize(250, 250)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid gray;")
            
            self.image_widgets[title] = image_label
            
            step_layout.addWidget(title_label)
            step_layout.addWidget(image_label)
            self.image_grid_layout.addLayout(step_layout)
        
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        
        image_display_layout.addWidget(scroll_area)
        main_layout.addLayout(image_display_layout, 4)

    # --- Fungsi Utilitas ---

    def convert_cv_to_qt(self, cv_img, color_fmt=cv2.COLOR_BGR2RGB):
        """Mengubah citra OpenCV ke format QPixmap."""
        if cv_img is None:
            return QPixmap()
            
        if len(cv_img.shape) == 3:
            # Citra Berwarna
            cv_img = cv2.cvtColor(cv_img, color_fmt)
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            # Citra Grayscale/Biner
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Skalakan citra agar sesuai dengan ukuran QLabel
        return QPixmap.fromImage(q_img).scaled(250, 250, Qt.KeepAspectRatio)

    def update_image_display(self, title, img):
        """Memperbarui QLabel dengan citra hasil pemrosesan."""
        pixmap = self.convert_cv_to_qt(img)
        self.image_widgets[title].setPixmap(pixmap)
        self.image_widgets[title].setText("") # Hapus teks placeholder

    # --- Fungsi Input Gambar ---

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                   "Image Files (*.png *.jpg *.jpeg);;All Files (*)", 
                                                   options=options)
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is not None:
                self.update_image_display("A. Input Citra", self.original_image)
                self.btn_process.setEnabled(True)
                self.result_text.setText("Gambar siap diproses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")
                self.btn_process.setEnabled(False)
    
    # --- FUNGSI ALGORITMA PCD UTAMA ---

    def process_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Perhatian", "Mohon input gambar terlebih dahulu.")
            return

        # 1. Pre-processing: Konversi ke Grayscale dan Filter
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Terapkan filter median untuk mengurangi noise
        preprocessed_img = cv2.medianBlur(gray_img, 5)
        self.update_image_display("B. Pre-processing (Gray)", preprocessed_img)

        # 2. Segmentasi Warna BERBASIS HSV
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Ikan Kerapu Sunu sering memiliki bintik biru/merah. Kita targetkan warna cerah.
        # Target: Warna dasar kemerahan/oranye atau bintik biru (tergantung variasi citra)
        
        # --- Contoh Range HSV untuk Warna Merah/Oranye (Warna Umum Kerapu Sunu) ---
        # Note: Range ini mungkin perlu disesuaikan dengan kondisi citra Anda.
        
        # Range Merah (Hue rendah dan tinggi karena merah melingkar)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        final_mask = mask1 + mask2
        
        # --- Tambahkan Filter Morfologi ---
        kernel = np.ones((5, 5), np.uint8)
        # Erosi diikuti Dilasi (Open) untuk menghilangkan noise kecil
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        # Dilasi untuk menyambung area yang terfragmentasi
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        self.update_image_display("C. Segmentasi Warna (Mask HSV)", final_mask)

        # 3. Ekstraksi Fitur dan Analisis
        
        # Temukan kontur
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter berdasarkan ukuran area minimum
            if area < 500: # Amang batas area minimum, sesuaikan
                continue
            
            # Ekstraksi Fitur Bentuk (Shape)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            
            # Hitung 'Circularity' (Rasio Area terhadap Perimeter^2, semakin dekat ke 1 semakin bulat)
            # Kerapu Sunu cenderung memanjang, jadi circularity-nya rendah
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Ekstraksi Fitur Tekstur (GLCM) - Perlu citra abu-abu dari area kontur
            x, y, w, h = cv2.boundingRect(contour)
            object_roi = preprocessed_img[y:y+h, x:x+w]
            
            # --- ATURAN KLASIK BERBASIS FITUR ---
            # 1. Bentuk (Circularity): Kerapu sunu memanjang (circularity rendah).
            # 2. Aspect Ratio (Rasio Lebar/Tinggi): Kerapu sunu cenderung lebih panjang (ratio > 1).
            # 3. Area: Ukuran yang wajar.
            
            aspect_ratio = float(w) / h
            
            # Amang batas (threshold) yang sangat disederhanakan:
            # Perlu disesuaikan melalui eksperimen pada dataset Anda!
            
            if (0.1 < circularity < 0.5) and (aspect_ratio > 0.8) and (area > 1000):
                is_kerapu_sunu_detected = True
                
                # Gambar Bounding Box di hasil akhir
                cv2.rectangle(detected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Teks
                cv2.putText(detected_img, f'KERAPU SUNU ({area:.0f})', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 4. Tampilkan Hasil Akhir dan Analisis
        self.update_image_display("D. Hasil Deteksi", detected_img)
        
        if is_kerapu_sunu_detected:
            self.result_text.setText("✅ DETEKSI BERHASIL: Objek Ikan Kerapu Sunu ditemukan!")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        else:
            self.result_text.setText("❌ DETEKSI GAGAL: Objek Kerapu Sunu tidak ditemukan.")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    detector = KerapuSunuDetector()
    detector.show()
    sys.exit(app.exec_())