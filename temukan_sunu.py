import sys
import cv2
import numpy as np
import base64
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QScrollArea, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

class KerapuSunuDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Ikan Kerapu Sunu (DFT/FFT Analysis)")
        self.setGeometry(100, 100, 1600, 850) 
        
        self.original_image = None
        self.processed_step_b = None 
        self.processed_step_c = None 
        self.processed_step_d = None 
        self.processed_step_e = None # **Magnitude Spectrum Visual (DFT)**
        self.detected_img = None     
        self.result_text_string = "Silakan Input Gambar dan Proses Deteksi."
        self.high_freq_energy = 0 
        
        self.init_ui()

    # --- Bagian UI dan Utilities (SAMA) ---

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        self.btn_input = QPushButton("1. Input Gambar")
        self.btn_input.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_input)
        
        self.btn_process = QPushButton("2. Proses Deteksi")
        self.btn_process.clicked.connect(self.process_detection)
        self.btn_process.setEnabled(False) 
        control_layout.addWidget(self.btn_process)
        
        self.btn_report = QPushButton("3. Generate Laporan (HTML)")
        self.btn_report.clicked.connect(self.generate_report)
        self.btn_report.setEnabled(False) 
        control_layout.addWidget(self.btn_report)
        
        control_layout.addStretch(1)
        
        self.result_label = QLabel("### Hasil Analisis:")
        self.result_text = QLabel(self.result_text_string)
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("font-size: 14pt; font-weight: bold; color: navy;")
        
        control_layout.addWidget(self.result_label)
        control_layout.addWidget(self.result_text)
        
        control_layout.addStretch(3)
        main_layout.addWidget(control_widget, 1)

        self.image_widgets = {}
        image_titles = [
            "A. Input Citra (RGB)", 
            "B. Pre-processing (Grayscale)", 
            "C. Segmentasi Ikan (Mask Biner)", 
            "D. Segmentasi Ikan (Masked)",
            "E. Spektrum Frekuensi (DFT)", # Tampilan Gambar DFT
            "F. Hasil Deteksi Akhir"
        ]
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.image_grid_layout = QHBoxLayout(scroll_content)
        
        for title in image_titles:
            step_layout = QVBoxLayout()
            title_label = QLabel(f"### {title}")
            title_label.setAlignment(Qt.AlignCenter)
            
            image_label = QLabel("Tidak Ada Gambar")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(150, 150)
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
            image_label.setStyleSheet("border: 1px solid gray;")
            
            self.image_widgets[title] = image_label
            step_layout.addWidget(title_label)
            step_layout.addWidget(image_label)
            self.image_grid_layout.addLayout(step_layout, 1)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 6)

    def convert_cv_to_qt(self, cv_img, color_fmt=cv2.COLOR_BGR2RGB):
        if cv_img is None: return QPixmap()
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, color_fmt)
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            if cv_img.dtype != np.uint8:
                cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def cv_to_base64(self, cv_img):
        if cv_img is None: return ""
        if len(cv_img.shape) == 3:
            _, buffer = cv2.imencode('.jpg', cv_img)
            mime_type = "image/jpeg"
        else:
            if cv_img.dtype != np.uint8:
                cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, buffer = cv2.imencode('.png', cv_img)
            mime_type = "image/png"
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:{mime_type};base64,{base64_string}"

    def update_image_display(self, title, img):
        if img is None:
            self.image_widgets[title].setPixmap(QPixmap())
            self.image_widgets[title].setText("Tidak Ada Gambar")
            return
        pixmap = self.convert_cv_to_qt(img)
        label = self.image_widgets[title]
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ) 
        label.setPixmap(scaled_pixmap)
        label.setText("")

    def update_all_processed_images(self):
        if self.original_image is not None:
            self.update_image_display("A. Input Citra (RGB)", self.original_image)
        if self.processed_step_b is not None:
            self.update_image_display("B. Pre-processing (Grayscale)", self.processed_step_b)
        if self.processed_step_c is not None:
            self.update_image_display("C. Segmentasi Ikan (Mask Biner)", self.processed_step_c)
        if self.processed_step_d is not None:
            self.update_image_display("D. Segmentasi Ikan (Masked)", self.processed_step_d)
        if self.processed_step_e is not None:
            self.update_image_display("E. Spektrum Frekuensi (DFT)", self.processed_step_e)
        if self.detected_img is not None:
             self.update_image_display("F. Hasil Deteksi Akhir", self.detected_img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_all_processed_images()

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                   "Image Files (*.png *.jpg *.jpeg);;All Files (*)", 
                                                   options=options)
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.processed_step_b = self.processed_step_c = self.processed_step_d = self.processed_step_e = self.detected_img = None
                self.update_all_processed_images()
                self.update_image_display("A. Input Citra (RGB)", self.original_image)
                self.btn_process.setEnabled(True)
                self.btn_report.setEnabled(False)
                self.result_text.setText("Gambar siap diproses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")
                self.btn_process.setEnabled(False)

    # --- FUNGSI ALGORITMA PCD UTAMA (Deteksi DFT) ---

    def process_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Perhatian", "Mohon input gambar terlebih dahulu.")
            return
            
        # --- KONSTANTA DFT ---
        # Ambang batas ini mungkin perlu diuji coba (disesuaikan dengan hasil energi yang Anda dapatkan)
        HIGH_FREQ_ENERGY_THRESHOLD = 5000000000 
        # ---------------------
        
        MIN_FISH_CONTOUR_AREA = 5000 

        # 1. Pre-processing (Grayscale)
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        preprocessed_img = cv2.medianBlur(gray_img, 5)
        self.processed_step_b = preprocessed_img.copy() 
        self.update_image_display("B. Pre-processing (Grayscale)", self.processed_step_b)

        # 2. Segmentasi Warna Ikan (Mask Biner)
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        final_mask_ikan = mask1 + mask2
        
        kernel = np.ones((5, 5), np.uint8)
        final_mask_ikan = cv2.morphologyEx(final_mask_ikan, cv2.MORPH_OPEN, kernel)
        final_mask_ikan = cv2.dilate(final_mask_ikan, kernel, iterations=1)
        
        # Menutup Lubang Hitam pada Mask Ikan
        kernel_close = np.ones((10, 10), np.uint8)
        final_mask_ikan = cv2.morphologyEx(final_mask_ikan, cv2.MORPH_CLOSE, kernel_close) 

        self.processed_step_c = final_mask_ikan.copy() 
        self.update_image_display("C. Segmentasi Ikan (Mask Biner)", self.processed_step_c)

        # D. Segmentasi Ikan (Masked)
        masked_fish = cv2.bitwise_and(self.original_image, self.original_image, mask=final_mask_ikan)
        self.processed_step_d = masked_fish.copy()
        self.update_image_display("D. Segmentasi Ikan (Masked)", self.processed_step_d)


        # 3. Ekstraksi Fitur Bentuk dan Analisis DFT
        contours, _ = cv2.findContours(final_mask_ikan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()
        self.high_freq_energy = 0 
        area_of_detected_fish = 0

        # Inisialisasi visualisasi spektrum (default hitam)
        self.processed_step_e = np.zeros_like(self.processed_step_b)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_FISH_CONTOUR_AREA: continue 
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            x, y, w, h = cv2.boundingRect(contour) 
            aspect_ratio = float(w) / h
            
            is_shape_ok = (0.1 < circularity < 0.5) and (aspect_ratio > 0.8)

            if is_shape_ok:
                
                # --- ANALISIS TEKSTUR DENGAN DFT ---
                
                # 1. Ambil ROI Grayscale dan Apply Mask Ikan
                gray_roi_full = self.processed_step_b[y:y+h, x:x+w]
                body_mask_roi = final_mask_ikan[y:y+h, x:x+w]
                
                gray_roi_masked = cv2.bitwise_and(gray_roi_full, gray_roi_full, mask=body_mask_roi)

                # 2. Hitung DFT
                rows, cols = gray_roi_masked.shape
                crow, ccol = rows//2 , cols//2
                
                f = np.fft.fft2(gray_roi_masked)
                fshift = np.fft.fftshift(f)
                
                # Hitung Spektrum Magnitudo (untuk visualisasi)
                magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)
                
                # 3. Visualisasi Spektrum Magnitudo (Kolom E)
                magnitude_display = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_step_e = magnitude_display.astype(np.uint8)

                # 4. Hitung Energi Frekuensi Tinggi
                mask = np.zeros((rows, cols), np.uint8)
                r = 20 # Radius lingkaran frekuensi rendah
                cv2.circle(mask, (ccol, crow), r, 1, -1)
                
                fshift_low = fshift * mask
                
                high_freq_fshift = fshift - fshift_low
                
                # Hitung Energi: Sum dari Magnitudo Kuadrat Frekuensi Tinggi
                high_freq_magnitude = np.abs(high_freq_fshift)
                current_energy = np.sum(high_freq_magnitude**2)
                
                self.high_freq_energy = current_energy
                
                # --- Kriteria Deteksi Akhir: Bentuk + Energi DFT ---
                is_texture_ok = current_energy > HIGH_FREQ_ENERGY_THRESHOLD
                
                if is_shape_ok and is_texture_ok:
                    is_kerapu_sunu_detected = True
                    
                    # Gambar Bounding Box Ikan di Hasil Akhir (F)
                    cv2.rectangle(detected_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    
                    # Teks hasil DFT
                    cv2.putText(detected_img, f'K. SUNU (DFT E:{current_energy:.2e})', (int(x), int(y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 4. Tampilkan Hasil Akhir dan Analisis
        
        # --- Bagian yang memastikan gambar DFT ditampilkan ---
        self.update_image_display("E. Spektrum Frekuensi (DFT)", self.processed_step_e)
        # ---------------------------------------------------
        
        self.detected_img = detected_img.copy() 
        self.update_image_display("F. Hasil Deteksi Akhir", self.detected_img)
        
        if is_kerapu_sunu_detected:
            self.result_text_string = f"DETEKSI BERHASIL: Objek Kerapu Sunu ditemukan (Bentuk cocok dan Energi DFT ({self.high_freq_energy:.2e}) melebihi ambang batas {HIGH_FREQ_ENERGY_THRESHOLD:.2e})."
            self.result_text.setText(f"✅ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        else:
            self.result_text_string = f"DETEKSI GAGAL: Objek Kerapu Sunu tidak ditemukan (Energi DFT ({self.high_freq_energy:.2e}) di bawah ambang batas {HIGH_FREQ_ENERGY_THRESHOLD:.2e} atau bentuk tidak cocok)."
            self.result_text.setText(f"❌ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;")
            
        self.btn_report.setEnabled(True)

    # --- Fungsi Generate Laporan HTML (SAMA) ---

    def generate_report(self):
        if self.original_image is None or self.detected_img is None:
            QMessageBox.warning(self, "Perhatian", "Mohon proses deteksi terlebih dahulu.")
            return

        HIGH_FREQ_ENERGY_THRESHOLD = 5000000000
        
        img_a = self.cv_to_base64(self.original_image)
        img_b = self.cv_to_base64(self.processed_step_b)
        img_c = self.cv_to_base64(self.processed_step_c) 
        img_d = self.cv_to_base64(self.processed_step_d) 
        img_e = self.cv_to_base64(self.processed_step_e) 
        img_f = self.cv_to_base64(self.detected_img)     
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Proyek Deteksi Kerapu Sunu (DFT/FFT Analysis)</title>
            <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                /* CSS SAMA */
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Laporan Proyek Akhir Mata Kuliah PCD: Deteksi Ikan Kerapu Sunu</h1>
                <p><strong>Tanggal Proses:</strong> {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</p>
                
                <div class="step">
                    <h2>1. Input Citra (A)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Citra RGB awal diakuisisi.</div>
                        <div class="image-container"><img src="{img_a}" alt="Input Citra"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>2. Pre-processing (Grayscale) (B)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Citra dikonversi ke **Grayscale** dan diterapkan **Filter Median 5x5**.</div>
                        <div class="image-container"><img src="{img_b}" alt="Pre-processing Grayscale"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>3. Segmentasi Ikan (Mask Biner) (C)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask Biner dihasilkan dari thresholding HSV. Operasi **Close** diterapkan untuk menutup lubang hitam, memastikan kontinuitas mask ikan.</div>
                        <div class="image-container"><img src="{img_c}" alt="Segmentasi Mask Biner Ikan"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>4. Segmentasi Ikan (Masked) (D)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask Biner diterapkan pada citra RGB asli. ROI ikan (Gray) ini digunakan untuk analisis DFT.</div>
                        <div class="image-container"><img src="{img_d}" alt="Segmentasi Ikan Masked"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>5. Spektrum Frekuensi (DFT) (E)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Analisa:</strong> DFT/FFT dilakukan pada ROI ikan Grayscale. **Spektrum Magnitudo** (diperlihatkan di atas) menunjukkan distribusi energi: Pusat spektrum adalah frekuensi rendah (bentuk), dan pinggiran adalah frekuensi tinggi (detail/bintik).
                        </div>
                        <div class="image-container"><img src="{img_e}" alt="Spektrum Magnitudo DFT"></div>
                    </div>
                </div>
                
                <div class="step">
                    <h2>6. Hasil Deteksi Akhir (F)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Hasil Akhir:</strong> {self.result_text_string} <br><br>
                            <strong>Kriteria Deteksi:</strong> 
                            <ul>
                                <li>**Bentuk:** Circularity & Aspect Ratio cocok.</li>
                                <li>**Tekstur (DFT):** Dihitung total **Energi Frekuensi Tinggi**. Jika energi melebihi ambang batas **{HIGH_FREQ_ENERGY_THRESHOLD:.2e}**, ikan dianggap berbintik.</li>
                            </ul>
                        </div>
                        <div class="image-container"><img src="{img_f}" alt="Hasil Deteksi Akhir"></div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan HTML", "Laporan_Deteksi_Kerapu_Sunu_DFT.html",
                                                   "HTML Files (*.html);;All Files (*)", options=options)
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(html_content)
                QMessageBox.information(self, "Sukses", f"Laporan berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan file: {e}")

if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    detector = KerapuSunuDetector()
    detector.show()
    sys.exit(app.exec_())