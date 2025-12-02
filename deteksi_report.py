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
        self.setWindowTitle("Deteksi Ikan Kerapu Sunu (PCD Klasik) - Proyek Akhir")
        self.setGeometry(100, 100, 1400, 800) 
        
        self.original_image = None
        self.processed_step_b = None
        self.processed_step_c = None
        self.detected_img = None
        self.result_text_string = "Silakan Input Gambar dan Proses Deteksi."
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # --- Kolom 1: Kontrol dan Hasil (Responsif: Stretch 1) ---
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
        main_layout.addWidget(control_widget, 1) # Stretch 1

        # --- Kolom 2: Tampilan Proses (Responsif: Stretch 4) ---
        
        self.image_widgets = {}
        image_titles = ["A. Input Citra", "B. Pre-processing (Gray)", 
                        "C. Segmentasi Warna (Mask HSV)", "D. Hasil Deteksi"]
        
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
            image_label.setMinimumSize(150, 150) # Ukuran minimum awal
            # Set policy agar label mengembang/menyusut sesuai ruang
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
            image_label.setStyleSheet("border: 1px solid gray;")
            
            self.image_widgets[title] = image_label
            
            step_layout.addWidget(title_label)
            step_layout.addWidget(image_label)
            self.image_grid_layout.addLayout(step_layout, 1) # Setiap gambar memiliki stretch 1
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 4) # Stretch 4

    # --- UTILITIES ---

    def convert_cv_to_qt(self, cv_img, color_fmt=cv2.COLOR_BGR2RGB):
        """Mengubah citra OpenCV ke format QPixmap."""
        if cv_img is None:
            return QPixmap()
            
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, color_fmt)
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return QPixmap.fromImage(q_img)

    def cv_to_base64(self, cv_img):
        """Mengkonversi citra OpenCV ke string Base64 untuk HTML."""
        if cv_img is None:
            return ""
        
        if len(cv_img.shape) == 3:
            _, buffer = cv2.imencode('.jpg', cv_img)
            mime_type = "image/jpeg"
        else:
            _, buffer = cv2.imencode('.png', cv_img)
            mime_type = "image/png"

        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:{mime_type};base64,{base64_string}"

    def update_image_display(self, title, img):
        """Memperbarui QLabel dengan citra hasil pemrosesan (Skala Responsif)."""
        if img is None:
            self.image_widgets[title].setPixmap(QPixmap())
            self.image_widgets[title].setText("Tidak Ada Gambar")
            return
            
        pixmap = self.convert_cv_to_qt(img)
        label = self.image_widgets[title]
        
        # Skalakan QPixmap agar sesuai dengan ukuran label saat ini
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ) 
        
        label.setPixmap(scaled_pixmap)
        label.setText("")

    def update_all_processed_images(self):
        """Memperbarui semua gambar yang sudah diproses saat jendela diubah ukurannya."""
        if self.original_image is not None:
            self.update_image_display("A. Input Citra", self.original_image)
        if self.processed_step_b is not None:
            self.update_image_display("B. Pre-processing (Gray)", self.processed_step_b)
        if self.processed_step_c is not None:
            self.update_image_display("C. Segmentasi Warna (Mask HSV)", self.processed_step_c)
        if self.detected_img is not None:
             self.update_image_display("D. Hasil Deteksi", self.detected_img)

    def resizeEvent(self, event):
        """Mengelola event perubahan ukuran jendela untuk menjaga responsivitas gambar."""
        super().resizeEvent(event)
        self.update_all_processed_images()

    # --- FUNGSI UTAMA ---

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                   "Image Files (*.png *.jpg *.jpeg);;All Files (*)", 
                                                   options=options)
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is not None:
                # Reset status dan gambar lama
                self.processed_step_b = self.processed_step_c = self.detected_img = None
                self.update_all_processed_images()
                
                self.update_image_display("A. Input Citra", self.original_image)
                self.btn_process.setEnabled(True)
                self.btn_report.setEnabled(False)
                self.result_text.setText("Gambar siap diproses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")
                self.btn_process.setEnabled(False)

    def process_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Perhatian", "Mohon input gambar terlebih dahulu.")
            return

        # 1. Pre-processing: Grayscale dan Filter Median
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        preprocessed_img = cv2.medianBlur(gray_img, 5)
        self.processed_step_b = preprocessed_img.copy() # Simpan
        self.update_image_display("B. Pre-processing (Gray)", self.processed_step_b)

        # 2. Segmentasi Warna BERBASIS HSV
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Range Merah/Oranye (Warna dasar Kerapu Sunu)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        
        final_mask = mask1 + mask2
        
        # Operasi Morfologi (Open & Dilate)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        self.processed_step_c = final_mask.copy() # Simpan
        self.update_image_display("C. Segmentasi Warna (Mask HSV)", self.processed_step_c)

        # 3. Ekstraksi Fitur dan Analisis
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000: # Filter area kecil
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            
            # Circularity: Rasio Area terhadap Perimeter^2 (0-1, semakin rendah semakin memanjang)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # --- ATURAN DETEKSI KLASIK ---
            # Perlu disesuaikan dengan dataset nyata!
            if (0.1 < circularity < 0.5) and (aspect_ratio > 0.8) and (area > 2000):
                is_kerapu_sunu_detected = True
                
                cv2.rectangle(detected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(detected_img, f'KERAPU SUNU ({circularity:.2f})', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 4. Tampilkan Hasil Akhir dan Analisis
        self.detected_img = detected_img.copy() 
        self.update_image_display("D. Hasil Deteksi", self.detected_img)
        
        if is_kerapu_sunu_detected:
            self.result_text_string = "DETEKSI BERHASIL: Objek Ikan Kerapu Sunu ditemukan berdasarkan fitur bentuk dan warna yang cocok."
            self.result_text.setText(f"✅ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        else:
            self.result_text_string = "DETEKSI GAGAL: Objek Kerapu Sunu tidak ditemukan. Fitur yang diekstraksi tidak memenuhi kriteria bentuk/warna yang ditentukan."
            self.result_text.setText(f"❌ {self.result_text_string}")
            self.result_text.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;")
            
        self.btn_report.setEnabled(True)

    def generate_report(self):
        if self.original_image is None or self.detected_img is None:
            QMessageBox.warning(self, "Perhatian", "Mohon proses deteksi terlebih dahulu.")
            return

        # Konversi semua citra yang diperlukan ke Base64
        img_a = self.cv_to_base64(self.original_image)
        img_b = self.cv_to_base64(self.processed_step_b)
        img_c = self.cv_to_base64(self.processed_step_c)
        img_d = self.cv_to_base64(self.detected_img)
        
        # Susun Konten Laporan HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Proyek Deteksi Kerapu Sunu (PCD Klasik)</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #004d40; border-bottom: 2px solid #004d40; padding-bottom: 10px; }}
                h2 {{ color: #00695c; margin-top: 30px; }}
                .step {{ border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .step img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #aaa; }}
                .analysis {{ background-color: #e0f2f1; padding: 10px; border-left: 5px solid #004d40; }}
            </style>
        </head>
        <body>
            <h1>Laporan Proyek Akhir Mata Kuliah PCD: Deteksi Ikan Kerapu Sunu</h1>
            <p><strong>Tanggal Proses:</strong> {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</p>
            
            <div class="step">
                <h2>1. Input Citra (A)</h2>
                <div class="analysis">
                    <strong>Analisa:</strong> Citra RGB awal diakuisisi. Langkah selanjutnya adalah menyiapkan citra untuk segmentasi dengan mengurangi noise dan mengonversi ke ruang warna yang lebih stabil (HSV).
                </div>
                <img src="{img_a}" alt="Input Citra">
            </div>

            <div class="step">
                <h2>2. Pre-processing (Grayscale & Filter Median) (B)</h2>
                <div class="analysis">
                    <strong>Analisa:</strong> Citra dikonversi ke **Grayscale** dan diterapkan **Filter Median 5x5**. Filter ini mereduksi *noise* bawah air tanpa mengaburkan batas objek secara berlebihan.
                </div>
                <img src="{img_b}" alt="Pre-processing Grayscale">
            </div>

            <div class="step">
                <h2>3. Segmentasi Warna (Mask HSV) (C)</h2>
                <div class="analysis">
                    <strong>Analisa:</strong> Citra diolah di ruang warna **HSV**. Thresholding digunakan untuk mengisolasi warna merah/oranye. Operasi **Morfologi (Open dan Dilasi)** diterapkan untuk membersihkan mask dan mempersiapkan kontur objek.
                </div>
                <img src="{img_c}" alt="Segmentasi Mask HSV">
            </div>

            <div class="step">
                <h2>4. Hasil Deteksi dan Ekstraksi Fitur (D)</h2>
                <div class="analysis">
                    <strong>Hasil Akhir:</strong> {self.result_text_string} <br>
                    <strong>Metode Analisa:</strong> Deteksi berbasis **aturan klasik** menggunakan **Area**, **Circularity**, dan **Aspect Ratio**. Objek yang memenuhi kriteria bentuk Kerapu Sunu ditandai dengan kotak hijau.
                </div>
                <img src="{img_d}" alt="Hasil Deteksi">
            </div>

        </body>
        </html>
        """
        
        # Simpan File HTML
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan HTML", "Laporan_Deteksi_Kerapu_Sunu.html",
                                                   "HTML Files (*.html);;All Files (*)", options=options)
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(html_content)
                QMessageBox.information(self, "Sukses", f"Laporan berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan file: {e}")

if __name__ == '__main__':
    # Pengaturan untuk layar high-DPI agar gambar tidak terlihat terlalu kecil
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    detector = KerapuSunuDetector()
    detector.show()
    sys.exit(app.exec_())