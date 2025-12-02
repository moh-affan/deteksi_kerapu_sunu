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
        self.setWindowTitle("Deteksi Kerapu Sunu (Tahap Lanjut: Segmentasi Objek)")
        # Menyesuaikan lebar untuk 7 kolom
        self.setGeometry(100, 100, 1800, 850) 
        
        self.original_image = None
        self.processed_step_a = None # Mask Otsu (Segmentasi Objek Awal)
        self.processed_step_b = None # Mask Otsu dengan Fill Holes
        self.processed_step_c = None # Mask Warna Ikan (Final)
        self.processed_step_d = None # Segmentasi Ikan (Masked)
        self.processed_step_e = None # Bintik Terdeteksi (Visual)
        self.detected_img = None     
        self.result_text_string = "Silakan Input Gambar dan Proses Deteksi."
        self.total_spot_area_detected = 0
        self.current_spot_percent = 0.0 
        
        self.init_ui()

    # --- Bagian UI dan Utilities (Disesuaikan untuk 7 Langkah) ---

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Kolom Kontrol (Sama)
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
        # 7 Slot Gambar
        image_titles = [
            "1. Input Citra (RGB)", 
            "2. Segmentation Awal",  # BARU A
            "3. Fill Holes (Mask Objek)",   # BARU B
            "4. Mask Warna Ikan (Final)",   # C (HSV filter applied to B)
            "5. Ikan Tersegmentasi",
            "6. Bintik Terdeteksi (Visual)",
            "7. Hasil Deteksi Akhir"
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
        main_layout.addWidget(scroll_area, 7) # Stretch 7 untuk 7 kolom

    # Metode Utilities (Konversi, Update, Resize, Load Image) - Disesuaikan untuk 7 slot
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
            self.update_image_display("1. Input Citra (RGB)", self.original_image)
        if self.processed_step_a is not None:
            self.update_image_display("2. Segmentation Awal", self.processed_step_a)
        if self.processed_step_b is not None:
            self.update_image_display("3. Fill Holes (Mask Objek)", self.processed_step_b)
        if self.processed_step_c is not None:
            self.update_image_display("4. Mask Warna Ikan (Final)", self.processed_step_c)
        if self.processed_step_d is not None:
            self.update_image_display("5. Ikan Tersegmentasi", self.processed_step_d)
        if self.processed_step_e is not None:
            self.update_image_display("6. Bintik Terdeteksi (Visual)", self.processed_step_e)
        if self.detected_img is not None:
             self.update_image_display("7. Hasil Deteksi Akhir", self.detected_img)

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
                self.processed_step_a = self.processed_step_b = self.processed_step_c = self.processed_step_d = self.processed_step_e = self.detected_img = None
                self.update_all_processed_images()
                self.update_image_display("1. Input Citra (RGB)", self.original_image)
                self.btn_process.setEnabled(True)
                self.btn_report.setEnabled(False)
                self.result_text.setText("Gambar siap diproses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")
                self.btn_process.setEnabled(False)

    # --- FUNGSI ALGORITMA PCD UTAMA (Diperbarui dengan Segmentasi Objek Awal) ---

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

        # 0. Pra-proses Umum
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)


        # # 1. TAHAP SEGMENTASI OBJEK AWAL (Otsu's Thresholding)
        # # Tujuannya: Mendapatkan mask kasar yang memisahkan foreground dari background
        # blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # # Ambang batas otomatis Otsu
        # _, initial_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # self.processed_step_a = initial_mask.copy() 
        # self.update_image_display("2. Segmentation Awal", self.processed_step_a)

        hue, s, v = cv2.split(hsv_img) # Pisahkan komponen V

        # 1. TAHAP SEGMENTASI OBJEK AWAL (Adaptive Thresholding pada V)
        # Memisahkan objek berdasarkan perubahan kecerahan lokal
        initial_mask = cv2.adaptiveThreshold(
            v, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, # Gunakan INVERSE agar foreground (ikan) menjadi putih
            25,                   # Ukuran blok (harus ganjil, lebih besar dari 1)
            10                    # Konstanta C yang dikurangi
        )
        # Aplikasikan Morfologi Open untuk membersihkan noise kecil
        kernel_open = np.ones((5, 5), np.uint8)
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open)

        self.processed_step_a = initial_mask.copy() 
        self.update_image_display("2. Segmentation Awal", self.processed_step_a)

        # 2. TAHAP PERBAIKAN MASK OBJEK (Fill Holes)
        # Menutup lubang pada mask Otsu (karena bintik terang dapat membuat lubang)
        mask_filled = initial_mask.copy()
        h, w = mask_filled.shape[:2]
        # Buat mask 2 piksel lebih besar (untuk operasi fill)
        mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
        # Lakukan Flood Fill dari (0,0) untuk mendapatkan background
        cv2.floodFill(mask_filled, mask_floodfill, (0, 0), 255)
        # Inversi mask: Background menjadi hitam, Foreground menjadi putih
        final_object_mask = cv2.bitwise_not(mask_filled) 

        self.processed_step_b = final_object_mask.copy() 
        self.update_image_display("3. Fill Holes (Mask Objek)", self.processed_step_b)


        # 3. TAHAP SEGMENTASI WARNA IKAN (Di dalam Mask Objek)
        
        # A. Filter HSV (Mencari Warna Ikan Merah/Oranye)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        hsv_color_mask = mask1 + mask2
        
        # B. Gabungkan: Objek harus memiliki Warna Ikan DAN berada di Mask Objek Awal
        final_mask_ikan = cv2.bitwise_and(hsv_color_mask, hsv_color_mask, mask=final_object_mask)
        
        # C. Morfologi akhir (menutup lubang kecil yang mungkin tercipta)
        kernel = np.ones((5, 5), np.uint8)
        final_mask_ikan = cv2.morphologyEx(final_mask_ikan, cv2.MORPH_CLOSE, kernel) 

        # 4. TAHAP EKSTRAKSI FITUR
        
        self.processed_step_c = final_mask_ikan.copy() 
        self.update_image_display("4. Mask Warna Ikan (Final)", self.processed_step_c)

        # 5. Tampilan Segmentasi Masked
        masked_fish = cv2.bitwise_and(self.original_image, self.original_image, mask=final_mask_ikan)
        self.processed_step_d = masked_fish.copy()
        self.update_image_display("5. Ikan Tersegmentasi", self.processed_step_d)


        # --- Analisis Bentuk dan Tekstur ---
        contours, _ = cv2.findContours(final_mask_ikan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()
        self.total_spot_area_detected = 0 
        area_of_detected_fish = 0
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
                
                # Deteksi Bintik (Sama seperti sebelumnya, tapi sekarang sangat terisolasi)
                required_min_total_spot_area = area * (MIN_TOTAL_SPOT_AREA_PERCENT / 100)
                required_max_area_per_spot = area * (MAX_AREA_PER_SPOT_PERCENT / 100)
                required_min_area_per_spot = area * (MIN_AREA_PER_SPOT_PERCENT / 100)
                
                # Deteksi Bintik Kecerahan Tinggi
                hsv_roi = hsv_img[y:y+h, x:x+w]
                hue, s, v = cv2.split(hsv_roi)
                v_roi = v.copy()
                
                spot_value_mask_adaptif = cv2.adaptiveThreshold(
                    v_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
                )
                spot_value_mask = cv2.medianBlur(spot_value_mask_adaptif, 3) 
                
                # Masking Ganda (Pastikan bintik ada di area ikan yang sudah tersegmentasi final)
                body_mask_roi = final_mask_ikan[y:y+h, x:x+w] 
                final_spot_mask = cv2.bitwise_and(spot_value_mask, body_mask_roi)
                
                spot_contours, _ = cv2.findContours(final_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                total_spot_area = 0
                
                for s_contour in spot_contours:
                    spot_area = cv2.contourArea(s_contour)
                    
                    if spot_area > required_min_area_per_spot and spot_area < required_max_area_per_spot:
                        total_spot_area += spot_area
                        
                        (sx, sy, sw, sh) = cv2.boundingRect(s_contour)
                        cv2.rectangle(spot_detection_visual, (int(x+sx), int(y+sy)), (int(x+sx+sw), int(y+sy+sh)), (0, 0, 255), 1)

                # --- Kriteria Deteksi Akhir ---
                self.total_spot_area_detected = total_spot_area
                area_of_detected_fish = area
                
                if total_spot_area > required_min_total_spot_area:
                    is_kerapu_sunu_detected = True
                    
                    cv2.rectangle(detected_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    
                    current_spot_percent = total_spot_area/area*100
                    self.current_spot_percent = current_spot_percent 
                    cv2.putText(detected_img, f'K. SUNU ({current_spot_percent:.2f}%)', (int(x), int(y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 6. Tampilan 6: Visualisasi Bintik Deteksi
        self.processed_step_e = spot_detection_visual.copy()
        self.update_image_display("6. Bintik Terdeteksi (Visual)", self.processed_step_e)

        # 7. Tampilan 7: Hasil Deteksi Akhir
        self.detected_img = detected_img.copy() 
        self.update_image_display("7. Hasil Deteksi Akhir", self.detected_img)
        
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

    # --- Fungsi Generate Laporan HTML (Disesuaikan untuk 7 Langkah) ---

    def generate_report(self):
        if self.original_image is None or self.detected_img is None:
            QMessageBox.warning(self, "Perhatian", "Mohon proses deteksi terlebih dahulu.")
            return

        MIN_TOTAL_SPOT_AREA_PERCENT = 1.0  
        MAX_AREA_PER_SPOT_PERCENT = 0.5    
        MIN_AREA_PER_SPOT_PERCENT = 0.01   
        
        result_text_final = getattr(self, 'result_text_string', 'Analisis tidak dijalankan.')
        current_spot_percent = getattr(self, 'current_spot_percent', 0.0)

        img_a = self.cv_to_base64(self.original_image)
        img_b = self.cv_to_base64(self.processed_step_a) 
        img_c = self.cv_to_base64(self.processed_step_b) 
        img_d = self.cv_to_base64(self.processed_step_c) 
        img_e = self.cv_to_base64(self.processed_step_d) 
        img_f = self.cv_to_base64(self.processed_step_e) 
        img_g = self.cv_to_base64(self.detected_img)     
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Deteksi Ikan Kerapu Sunu</title>
            <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                /* CSS SAMA */
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Laporan Proyek Akhir Mata Kuliah PCD: Deteksi Ikan Kerapu Sunu</h1>
                <p><strong>Waktu Proses:</strong> {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</p>
                <div class="final-result">
                    <strong>STATUS DETEKSI:</strong> {result_text_final}
                </div>
                
                <div class="step">
                    <h2>1. Input Citra (A)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Citra RGB masukan.</div>
                        <div class="image-container"><img src="{img_a}" alt="Input Citra"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>2. Segmentation Awal (Otsu) (B)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Langkah pertama segmentasi objek menggunakan **Otsu's Thresholding** pada Grayscale untuk mendapatkan mask foreground/background yang kasar.</div>
                        <div class="image-container"><img src="{img_b}" alt="Segmentation Awal Otsu"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>3. Fill Holes (Mask Objek) (C)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask dari Otsu diperbaiki menggunakan **Flood Fill** untuk menutup lubang (Fill Holes). Mask ini menjadi batasan (Constraint) sebelum filtering warna.</div>
                        <div class="image-container"><img src="{img_c}" alt="Fill Holes Mask Objek"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>4. Mask Warna Ikan (Final) (D)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> **Thresholding HSV** (Warna Ikan) diterapkan **DAN** digabungkan (bitwise AND) dengan Mask Objek (C). Operasi ini menghilangkan *noise* warna dari *background*.</div>
                        <div class="image-container"><img src="{img_d}" alt="Mask Warna Ikan Final"></div>
                    </div>
                </div>
                
                <div class="step">
                    <h2>5. Ikan Tersegmentasi (E)</h2>
                    <div class="step-content">
                        <div class="analysis"><strong>Analisa:</strong> Mask Warna Final (D) diterapkan pada citra RGB asli, menghasilkan ikan yang terisolasi dengan *background* hitam.</div>
                        <div class="image-container"><img src="{img_e}" alt="Ikan Tersegmentasi"></div>
                    </div>
                </div>

                <div class="step">
                    <h2>6. Deteksi Bintik (Visualisasi) (F)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Analisa:</strong> Bintik dideteksi dari piksel **Kecerahan Tinggi (Adaptive Threshold)** setelah *Masking Ganda*. Total area kontur bintik dihitung.
                            <br><br>
                            Total Area Bintik Terukur: <strong>{current_spot_percent:.2f}%</strong>.
                            <br>
                            Ambang Batas Minimum: <strong>{MIN_TOTAL_SPOT_AREA_PERCENT:.2f}%</strong>.
                        </div>
                        <div class="image-container"><img src="{img_f}" alt="Bintik Terdeteksi Visual"></div>
                    </div>
                </div>
                
                <div class="step">
                    <h2>7. Hasil Deteksi Akhir (G)</h2>
                    <div class="step-content">
                        <div class="analysis">
                            <strong>Kriteria Deteksi Final:</strong> 
                            <ul>
                                <li>**Bentuk:** Circularity & Aspect Ratio ikan harus cocok.</li>
                                <li>**Tekstur (Bintik):** Total area bintik terang harus melebihi ambang batas minimum.</li>
                            </ul>
                        </div>
                        <div class="image-container"><img src="{img_g}" alt="Hasil Deteksi Akhir"></div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan HTML", "Laporan_Deteksi_Kerapu_Sunu_Advanced.html",
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