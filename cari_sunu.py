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
        self.setWindowTitle("Deteksi Ikan Kerapu Sunu (Perbaikan Segmentasi)")
        self.setGeometry(100, 100, 1600, 850) 
        
        self.original_image = None
        self.processed_step_b = None 
        self.processed_step_c = None 
        self.processed_step_d = None 
        self.processed_step_e = None 
        self.detected_img = None     
        self.result_text_string = "Silakan Input Gambar dan Proses Deteksi."
        self.total_spot_area_detected = 0
        
        self.init_ui()

    # --- Bagian UI dan Utilities (Dibiarkan sama) ---

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
            "E. Bintik Terdeteksi (Visual)",
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
            self.update_image_display("E. Bintik Terdeteksi (Visual)", self.processed_step_e)
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

    # --- FUNGSI ALGORITMA PCD UTAMA (Diperbarui dengan Perbaikan Masking) ---

    def process_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Perhatian", "Mohon input gambar terlebih dahulu.")
            return
            
        # --- KONSTANTA PERSENTASE ---
        MIN_TOTAL_SPOT_AREA_PERCENT = 1.0  
        MAX_AREA_PER_SPOT_PERCENT = 0.5    
        MIN_AREA_PER_SPOT_PERCENT = 0.01   
        MIN_FISH_CONTOUR_AREA = 5000       
        # ----------------------------

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
        # Morfologi: OPEN (Hilangkan noise kecil), DILATE (Perluas sedikit)
        final_mask_ikan = cv2.morphologyEx(final_mask_ikan, cv2.MORPH_OPEN, kernel)
        final_mask_ikan = cv2.dilate(final_mask_ikan, kernel, iterations=1)
        
        # --- PERBAIKAN: Menutup Lubang Hitam pada Mask Ikan (Kolom C) ---
        kernel_close = np.ones((10, 10), np.uint8)
        final_mask_ikan = cv2.morphologyEx(final_mask_ikan, cv2.MORPH_CLOSE, kernel_close) 

        # C. Tampilan 1: Mask Biner
        self.processed_step_c = final_mask_ikan.copy() 
        self.update_image_display("C. Segmentasi Ikan (Mask Biner)", self.processed_step_c)

        # D. Tampilan 2: Segmentasi Ikan (Masked)
        # Terapkan mask biner ke gambar RGB asli
        masked_fish = cv2.bitwise_and(self.original_image, self.original_image, mask=final_mask_ikan)
        self.processed_step_d = masked_fish.copy()
        self.update_image_display("D. Segmentasi Ikan (Masked)", self.processed_step_d)


        # 3. Ekstraksi Fitur Bentuk dan Tekstur Persentase
        contours, _ = cv2.findContours(final_mask_ikan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()
        self.total_spot_area_detected = 0 
        area_of_detected_fish = 0

        # Inisialisasi visualisasi bintik
        spot_detection_visual = np.zeros_like(self.original_image)
        
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
                
                # Menghitung Threshold Dinamis dari Persentase
                required_min_total_spot_area = area * (MIN_TOTAL_SPOT_AREA_PERCENT / 100)
                required_max_area_per_spot = area * (MAX_AREA_PER_SPOT_PERCENT / 100)
                required_min_area_per_spot = area * (MIN_AREA_PER_SPOT_PERCENT / 100)
                
                
                # DETEKSI BINTIK BERDASARKAN KECERAHAN TINGGI (Value HSV)
                hsv_roi = hsv_img[y:y+h, x:x+w]
                hue, s, v = cv2.split(hsv_roi)
                
                # Threshold Kecerahan (V)
                _, spot_value_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
                spot_value_mask = cv2.medianBlur(spot_value_mask, 3) 
                
                # --- PERBAIKAN: Masking Ganda untuk memastikan bintik di dalam tubuh ikan ---
                # Ambil ROI dari mask ikan yang sudah ditutup lubangnya
                body_mask_roi = final_mask_ikan[y:y+h, x:x+w] 
                
                # Operasi AND: Bintik hanya valid jika terang DAN berada di area tubuh ikan
                final_spot_mask = cv2.bitwise_and(spot_value_mask, body_mask_roi)
                
                spot_contours, _ = cv2.findContours(final_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                total_spot_area = 0
                
                # Hitung Total Area Kontur Kecil (Bintik)
                for s_contour in spot_contours:
                    spot_area = cv2.contourArea(s_contour)
                    
                    if spot_area > required_min_area_per_spot and spot_area < required_max_area_per_spot:
                        total_spot_area += spot_area
                        
                        # Tandai bintik yang terdeteksi
                        (sx, sy, sw, sh) = cv2.boundingRect(s_contour)
                        
                        # Gambar bounding box merah kecil di lokasi bintik
                        cv2.rectangle(spot_detection_visual, (int(x+sx), int(y+sy)), (int(x+sx+sw), int(y+sy+sh)), (0, 0, 255), 1)

                # --- Kriteria Deteksi Akhir ---
                self.total_spot_area_detected = total_spot_area
                area_of_detected_fish = area
                
                if total_spot_area > required_min_total_spot_area:
                    is_kerapu_sunu_detected = True
                    
                    # Gambar Bounding Box Ikan di Hasil Akhir (F)
                    cv2.rectangle(detected_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    
                    current_spot_percent = total_spot_area/area*100
                    cv2.putText(detected_img, f'K. SUNU ({current_spot_percent:.2f}%)', (int(x), int(y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 4. Tampilkan Hasil Akhir dan Analisis
        
        # E. Tampilan 3: Visualisasi Bintik Deteksi
        self.processed_step_e = spot_detection_visual.copy()
        self.update_image_display("E. Bintik Terdeteksi (Visual)", self.processed_step_e)

        # F. Tampilan 4: Hasil Deteksi Akhir
        self.detected_img = detected_img.copy() 
        self.update_image_display("F. Hasil Deteksi Akhir", self.detected_img)
        
        current_spot_percent = (self.total_spot_area_detected / area_of_detected_fish * 100) if area_of_detected_fish > 0 else 0

        if is_kerapu_sunu_detected:
            self.result_text_string = f"DETEKSI BERHASIL: Bintik terang ({current_spot_percent:.2f}%) memenuhi kriteria {MIN_TOTAL_SPOT_AREA_PERCENT}%."
            self.result_text.setText(f"✅ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        else:
            self.result_text_string = f"DETEKSI GAGAL: Total area bintik terang ({current_spot_percent:.2f}%) di bawah {MIN_TOTAL_SPOT_AREA_PERCENT}% atau bentuk tidak cocok."
            self.result_text.setText(f"❌ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;")
            
        self.btn_report.setEnabled(True)

    # --- Fungsi Generate Laporan HTML (SAMA) ---

    def generate_report(self):
        if self.original_image is None or self.detected_img is None:
            QMessageBox.warning(self, "Perhatian", "Mohon proses deteksi terlebih dahulu.")
            return

        MIN_TOTAL_SPOT_AREA_PERCENT = 1.0  
        MAX_AREA_PER_SPOT_PERCENT = 0.5    
        MIN_AREA_PER_SPOT_PERCENT = 0.01   
        
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
            <title>Laporan Proyek Deteksi Kerapu Sunu (6 Langkah Visual)</title>
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
                        <div class="analysis"><strong>Analisa:</strong> Citra dikonversi ke **Grayscale** dan diterapkan **Filter Median 5x5** untuk mereduksi *noise*.</div>
                        <div class="image-container"><img src="{img_b}" alt="Pre-processing Grayscale"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>3. Segmentasi Ikan (Mask Biner) (C)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask Biner (putih = ikan, hitam = background) dihasilkan dari thresholding HSV. Operasi **Close** diterapkan untuk **menutup lubang hitam** pada tubuh ikan, memastikan kontinuitas.</div>
                        <div class="image-container"><img src="{img_c}" alt="Segmentasi Mask Biner Ikan"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>4. Segmentasi Ikan (Masked) (D)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask Biner diterapkan pada citra RGB asli untuk mengisolasi ikan dengan *background* hitam.</div>
                        <div class="image-container"><img src="{img_d}" alt="Segmentasi Ikan Masked"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>5. Bintik Terdeteksi (Visualisasi) (E)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Analisa:</strong> Bintik dideteksi dari piksel **Kecerahan Tinggi (Value > 200)**. **Masking Ganda (AND)** diterapkan dengan Mask Ikan (C) untuk memastikan **hanya bintik di dalam tubuh ikan** yang dihitung.
                        </div>
                        <div class="image-container"><img src="{img_e}" alt="Bintik Terdeteksi Visual"></div>
                    </div>
                </div>
                
                <div class="step">
                    <h2>6. Hasil Deteksi Akhir (F)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Hasil Akhir:</strong> {self.result_text_string} <br><br>
                            <strong>Kriteria Deteksi:</strong> Bentuk cocok **DAN** Total Area Bintik Terang melebihi **{MIN_TOTAL_SPOT_AREA_PERCENT}%** dari area ikan.
                        </div>
                        <div class="image-container"><img src="{img_f}" alt="Hasil Deteksi Akhir"></div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan HTML", "Laporan_Deteksi_Kerapu_Sunu_6Langkah.html",
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