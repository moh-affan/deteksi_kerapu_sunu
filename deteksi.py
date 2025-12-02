import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QScrollArea, QMessageBox, QSizePolicy)  # QSizePolicy sudah ada
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize  # <-- Tambahkan QSize di sini


class KerapuSunuDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Deteksi Ikan Kerapu Sunu (PCD Klasik) - Layout Responsif")
        # Mengatur ukuran awal, tetapi biarkan pengguna mengubah ukurannya
        self.setGeometry(100, 100, 1400, 800)

        self.original_image = None
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout menggunakan QHBoxLayout untuk membagi Kontrol dan Tampilan
        main_layout = QHBoxLayout(central_widget)

        # --- Kolom 1: Kontrol dan Hasil (Tetap Lebar Minimum) ---
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Tombol Input Gambar
        self.btn_input = QPushButton("1. Input Gambar")
        self.btn_input.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_input)

        # Tombol Proses
        self.btn_process = QPushButton("2. Proses Deteksi")
        self.btn_process.clicked.connect(self.process_detection)
        self.btn_process.setEnabled(False)
        control_layout.addWidget(self.btn_process)

        control_layout.addStretch(1)  # Stretch agar elemen ke bawah

        # Hasil Analisis Teks
        self.result_label = QLabel("### Hasil Analisis:")
        self.result_text = QLabel("Silakan Input Gambar.")
        self.result_text.setWordWrap(True)  # Agar teks panjang turun baris
        self.result_text.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: navy;")

        control_layout.addWidget(self.result_label)
        control_layout.addWidget(self.result_text)

        control_layout.addStretch(3)

        # Tambahkan Kontrol Widget ke Main Layout dengan stretch factor 1 (Lebar Minimum)
        main_layout.addWidget(control_widget, 1)

        # --- Kolom 2: Tampilan Proses (Image Viewer) ---

        self.image_widgets = {}
        image_titles = ["A. Input Citra", "B. Pre-processing (Gray)",
                        "C. Segmentasi Warna (Mask HSV)", "D. Hasil Deteksi"]

        # Gunakan QScrollArea untuk menampung gambar
        scroll_area = QScrollArea()
        # PENTING: Membuat area scroll dapat diubah ukurannya
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        # Menggunakan QHBoxLayout untuk tata letak gambar secara horizontal
        self.image_grid_layout = QHBoxLayout(scroll_content)

        for title in image_titles:
            step_layout = QVBoxLayout()
            title_label = QLabel(f"### {title}")
            title_label.setAlignment(Qt.AlignCenter)  # Pusatkan judul

            image_label = QLabel("Tidak Ada Gambar")
            image_label.setAlignment(Qt.AlignCenter)
            # Hapus fixed size agar label gambar responsif
            image_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,  # Kebijakan Horizontal
                QSizePolicy.Policy.Expanding  # Kebijakan Vertikal
            )
            image_label.setStyleSheet("border: 1px solid gray;")

            self.image_widgets[title] = image_label

            step_layout.addWidget(title_label)
            step_layout.addWidget(image_label)
            # Tambahkan layout gambar ke grid layout dengan stretch factor 1 (agar semua gambar berbagi ruang sama)
            self.image_grid_layout.addLayout(step_layout, 1)

        scroll_area.setWidget(scroll_content)

        # Tambahkan Scroll Area ke Main Layout dengan stretch factor 4 (Lebih lebar dari kontrol)
        main_layout.addWidget(scroll_area, 4)

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
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
        else:
            # Citra Grayscale/Biner
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height,
                           bytes_per_line, QImage.Format_Grayscale8)

        return QPixmap.fromImage(q_img)

    def update_image_display(self, title, img):
        """Memperbarui QLabel dengan citra hasil pemrosesan."""
        pixmap = self.convert_cv_to_qt(img)

        label = self.image_widgets[title]

        # Skalakan QPixmap agar sesuai dengan ukuran label (dan label sudah diatur agar mengembang)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(scaled_pixmap)
        label.setText("")  # Hapus teks placeholder

    # Mengatur ulang scaling saat ukuran jendela berubah
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # PENTING: Memastikan gambar di-skala ulang saat jendela diubah ukurannya
        if self.original_image is not None:
            # Kita hanya perlu memperbarui tampilan gambar terakhir yang telah diproses
            # Jika semua gambar harus diperbarui, lakukan loop di sini
            self.update_all_processed_images()

    def update_all_processed_images(self):
        """Fungsi pembantu untuk memperbarui semua gambar yang sudah diproses saat resize."""
        # Jika Anda ingin menyimpan semua citra hasil langkah proses (misalnya self.processed_step_c, etc)
        # maka Anda bisa memanggil update_image_display untuk setiap langkah di sini.

        # Karena kita hanya menyimpan original_image, kita akan membuat logika sederhana.
        # Untuk implementasi yang lebih baik, simpan hasil cv2 dari setiap langkah sebagai atribut kelas.

        # Contoh perbaikan sederhana untuk gambar A (Input Citra)
        if self.original_image is not None:
            self.update_image_display("A. Input Citra", self.original_image)
        # Jika deteksi sudah pernah dijalankan, panggil ulang prosesnya (ini kurang efisien)
        # Cara yang lebih baik: simpan hasil citra (misalnya self.detected_img) dan tampilkan ulang.

        # Asumsi: Jika deteksi sudah dijalankan, detected_img ada (misalnya disimpan sebagai atribut kelas)
        if hasattr(self, 'detected_img') and self.detected_img is not None:
            self.update_image_display("D. Hasil Deteksi", self.detected_img)
            # Ulangi untuk step B dan C jika Anda menyimpannya

    # ... (lanjutkan sisa kode load_image dan process_detection Anda di sini) ...
    # Pastikan di akhir process_detection, Anda menyimpan citra hasil: self.detected_img = detected_img.copy()

    # --- FUNGSI ALGORITMA PCD UTAMA (Sama dengan sebelumnya) ---

    def process_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Perhatian",
                                "Mohon input gambar terlebih dahulu.")
            return

        # 1. Pre-processing: Konversi ke Grayscale dan Filter
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        preprocessed_img = cv2.medianBlur(gray_img, 5)
        self.update_image_display("B. Pre-processing (Gray)", preprocessed_img)

        # Simpan hasil preprocessed
        self.processed_step_b = preprocessed_img

        # 2. Segmentasi Warna BERBASIS HSV
        hsv_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

        final_mask = mask1 + mask2

        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        self.update_image_display("C. Segmentasi Warna (Mask HSV)", final_mask)
        # Simpan hasil segmentasi
        self.processed_step_c = final_mask

        # 3. Ekstraksi Fitur dan Analisis
        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_kerapu_sunu_detected = False
        detected_img = self.original_image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if (0.1 < circularity < 0.5) and (aspect_ratio > 0.8) and (area > 1000):
                is_kerapu_sunu_detected = True

                cv2.rectangle(detected_img, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(detected_img, f'KERAPU SUNU ({area:.0f})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Tampilkan Hasil Akhir dan Analisis
        # Simpan citra hasil deteksi agar dapat di-skala ulang saat resize
        self.detected_img = detected_img.copy()
        self.update_image_display("D. Hasil Deteksi", self.detected_img)

        if is_kerapu_sunu_detected:
            self.result_text.setText(
                "✅ DETEKSI BERHASIL: Objek Ikan Kerapu Sunu ditemukan!")
            self.result_text.setStyleSheet(
                "font-size: 16pt; font-weight: bold; color: green;")
        else:
            self.result_text.setText(
                "❌ DETEKSI GAGAL: Objek Kerapu Sunu tidak ditemukan.")
            self.result_text.setStyleSheet(
                "font-size: 16pt; font-weight: bold; color: red;")

    def load_image(self):
        # ... (Kode load_image tetap sama) ...
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                   "Image Files (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)

        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)

            if self.original_image is not None:
                self.update_image_display(
                    "A. Input Citra", self.original_image)
                self.btn_process.setEnabled(True)
                self.result_text.setText("Gambar siap diproses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")
                self.btn_process.setEnabled(False)


if __name__ == '__main__':
    # Untuk memastikan scaling yang benar pada high-DPI screens
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    detector = KerapuSunuDetector()
    detector.show()
    sys.exit(app.exec_())
