import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton,
    QVBoxLayout, QWidget, QHBoxLayout, QSlider, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer

class ImageSegmenterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Color-Based Segmenter (PyQt5)")

        self.label = QLabel(self)

        self.label.setMinimumSize(800, 800)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setMouseTracking(True)



        self.undo_button = QPushButton("Undo Last Region")
        self.undo_button.clicked.connect(self.undo_last_region)

        self.save_button = QPushButton("Save Segmentation")
        self.save_button.clicked.connect(self.save_segmentation)

        self.h_slider = self.create_slider(10)
        self.s_slider = self.create_slider(20)
        self.v_slider = self.create_slider(20)

        self.h_value = QLabel("10")
        self.s_value = QLabel("20")
        self.v_value = QLabel("20")

        self.h_slider.valueChanged.connect(lambda v: self.h_value.setText(str(v)))
        self.s_slider.valueChanged.connect(lambda v: self.s_value.setText(str(v)))
        self.v_slider.valueChanged.connect(lambda v: self.v_value.setText(str(v)))

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.load_selected_file)

        self.folder_button = QPushButton("Open Folder")
        self.folder_button.clicked.connect(self.select_folder)

        slider_box = QGroupBox("HSV Tolerance")
        form_layout = QFormLayout()
        form_layout.addRow("File:", self.file_label)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.h_slider)
        h_layout.addWidget(self.h_value)
        form_layout.addRow("H Range", h_layout)

        s_layout = QHBoxLayout()
        s_layout.addWidget(self.s_slider)
        s_layout.addWidget(self.s_value)
        form_layout.addRow("S Range", s_layout)

        v_layout = QHBoxLayout()
        v_layout.addWidget(self.v_slider)
        v_layout.addWidget(self.v_value)
        form_layout.addRow("V Range", v_layout)

        slider_box.setLayout(form_layout)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.folder_button)
        control_layout.addWidget(self.file_list)
        control_layout.addWidget(slider_box)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.save_button)
        control_layout.addStretch()


        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setFixedWidth(320)  # or setMaximumWidth(320) if you want flexibility


        splitter = QSplitter()
        splitter.addWidget(self.label)
        splitter.addWidget(control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.image_original = None
        self.image_display = None
        self.scale_x = self.scale_y = 1.0
        self.file_path = ""

        self.selections = []
        self.masks = []
        self.start_pos = None
        self.current_rect = None

        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

        self.resize_timer = QTimer()
        self.resize_timer.setInterval(300)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.handle_resize)
        self.resizeEvent = self.delayed_resize_event

    def create_slider(self, default):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(default)
        slider.setTickInterval(5)
        return slider

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.tif', '.tiff')):
                    item = QListWidgetItem(fname)
                    item.setData(Qt.UserRole, os.path.join(folder, fname))
                    self.file_list.addItem(item)

    def load_selected_file(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.file_path = path
            self.image_original = cv2.imread(path)
            self.file_label.setText(os.path.basename(path))
            self.selections.clear()
            self.masks.clear()
            self.resize_image()
            self.update_display()

    def resize_image(self):
        if self.image_original is None:
            return
        label_width = self.label.width() if self.label.width() > 0 else 800
        label_height = self.label.height() if self.label.height() > 0 else 600

        h, w = self.image_original.shape[:2]
        scale = min(label_width / w, label_height / h)
        new_size = (int(w * scale), int(h * scale))
        self.image_display = cv2.resize(self.image_original, new_size)
        self.scale_x = w / new_size[0]
        self.scale_y = h / new_size[1]

    def update_display(self):
        if self.image_display is None:
            return

        if self.masks:
            mask = np.bitwise_or.reduce(self.masks)
        else:
            mask = np.zeros(self.image_original.shape[:2], dtype=np.uint8)

        green_overlay = np.zeros_like(self.image_original)
        green_overlay[:] = (0, 255, 0)
        blended = cv2.addWeighted(self.image_original, 1.0, green_overlay, 0.5, 0)
        overlay = np.where(mask[:, :, None] > 0, blended, self.image_original)
        overlay_resized = cv2.resize(overlay, (self.image_display.shape[1], self.image_display.shape[0]))
        image_rgb = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGB)
        qimage = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        if self.current_rect:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)
            painter.end()

        self.label.setPixmap(pixmap)

    def mouse_press(self, event):
        self.start_pos = event.pos()

    def mouse_move(self, event):
        if self.start_pos:
            self.current_rect = QRect(self.start_pos, event.pos())
            self.update_display()

    def mouse_release(self, event):
        if self.start_pos:
            x0 = int(min(self.start_pos.x(), event.x()) * self.scale_x)
            y0 = int(min(self.start_pos.y(), event.y()) * self.scale_y)
            x1 = int(max(self.start_pos.x(), event.x()) * self.scale_x)
            y1 = int(max(self.start_pos.y(), event.y()) * self.scale_y)
            if x1 > x0 and y1 > y0:
                self.selections.append((x0, y0, x1, y1))
                self.compute_mask_for_selection((x0, y0, x1, y1))
            self.start_pos = None
            self.current_rect = None
            self.update_display()

    def compute_mask_for_selection(self, selection):
        x0, y0, x1, y1 = selection
        region = self.image_original[y0:y1, x0:x1]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = cv2.mean(hsv_region)[:3]

        h_range = self.h_slider.value()
        s_range = self.s_slider.value()
        v_range = self.v_slider.value()

        h_region, w_region = y1 - y0, x1 - x0
        x0_local = max(0, x0 - 5 * w_region)
        y0_local = max(0, y0 - 5 * h_region)
        x1_local = min(self.image_original.shape[1], x1 + 5 * w_region)
        y1_local = min(self.image_original.shape[0], y1 + 5 * h_region)

        local_area = self.image_original[y0_local:y1_local, x0_local:x1_local]
        hsv_local = cv2.cvtColor(local_area, cv2.COLOR_BGR2HSV)

        lower = np.array([max(0, mean_h - h_range), max(0, mean_s - s_range), max(0, mean_v - v_range)], dtype=np.uint8)
        upper = np.array([min(179, mean_h + h_range), min(255, mean_s + s_range), min(255, mean_v + v_range)], dtype=np.uint8)

        local_mask = cv2.inRange(hsv_local, lower, upper)
        mask = np.zeros(self.image_original.shape[:2], dtype=np.uint8)
        mask[y0_local:y1_local, x0_local:x1_local] = local_mask
        self.masks.append(mask)

    def undo_last_region(self):
        if self.selections:
            self.selections.pop()
            self.masks.pop()
            self.update_display()

    def save_segmentation(self):
        if not self.masks or self.file_path == "":
            return
        final_mask = np.bitwise_or.reduce(self.masks)
        out_path = os.path.splitext(self.file_path)[0] + "_lab.png"
        mask_rgb = np.where(final_mask[:, :] > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(out_path, mask_rgb)
        print(f"Segmentation saved to {out_path}")

    def delayed_resize_event(self, event):
        self.resize_timer.start()
        super().resizeEvent(event)

    def handle_resize(self):
        self.resize_image()
        self.update_display()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSegmenterApp()
    window.resize(1400, 1000)
    window.show()
    sys.exit(app.exec_())