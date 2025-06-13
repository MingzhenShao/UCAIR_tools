import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QMessageBox, QGraphicsScene,
                             QGraphicsView, QGraphicsPixmapItem)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from aicspylibczi import CziFile
from tifffile import TiffFile

import cv2

class CZIMatcherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CZI Region Viewer")

        self.tiff_path = None
        self.png_path = None
        self.landmarks_path = None
        self.czi_path = None
        self.scale_ratio = None
        self.czi_array = None

        self.tiff_label = QLabel()
        self.tiff_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.tiff_label.mousePressEvent = self.on_click

        self.png_label = QLabel()
        self.png_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.region_label = QLabel()
        self.region_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)


        self.status_label = QLabel("Load PNG, CZI, and landmarks to start.")

        self.last_figure = None

        self.init_ui()

    def init_ui(self):
        btn_tiff = QPushButton("Load TIFF")
        btn_tiff.clicked.connect(self.load_tiff)

        btn_png = QPushButton("Load PNG")
        btn_png.clicked.connect(self.load_png)

        btn_landmarks = QPushButton("Load Landmarks")
        btn_landmarks.clicked.connect(self.load_landmarks)

        btn_czi = QPushButton("Load CZI")
        btn_czi.clicked.connect(self.load_czi)

        hbox = QHBoxLayout()
        hbox.addWidget(btn_tiff)
        hbox.addWidget(btn_png)
        hbox.addWidget(btn_landmarks)
        hbox.addWidget(btn_czi)

        image_hbox = QHBoxLayout()
        image_hbox.addWidget(self.tiff_label)
        image_hbox.addWidget(self.png_label)
        image_hbox.addWidget(self.region_label)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(image_hbox)
        vbox.addWidget(self.status_label)

        self.setLayout(vbox)


    def load_tiff(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select TIFF Image", filter="TIFF Files (*.tiff *.tif)")
        if path:
            self.tiff_path = path
            self.status_label.setText(f"TIFF loaded: {path}")
            self.try_initialize()

    def load_png(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PNG Image", filter="PNG Files (*.png)")
        if path:
            self.png_path = path
            self.status_label.setText(f"PNG loaded: {path}")
            self.try_initialize()

    def load_landmarks(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Landmarks File", filter="NumPy Files (*.npy)")
        if path:
            self.landmarks_path = path
            self.status_label.setText(f"Landmarks loaded: {path}")
            self.try_initialize()

    def load_czi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", filter="CZI and TIFF Files (*.czi *.tif *.tiff);;All Files (*)")
        if path:
            self.czi_path = path
            self.status_label.setText(f"CZI loaded: {path}")
            self.try_initialize()

    def try_initialize(self):
        if self.tiff_path and self.png_path and self.landmarks_path and self.czi_path:
            self.init_gui()

    def init_gui(self):
        self.tiff_image = Image.open(self.tiff_path)
        self.png_image = Image.open(self.png_path)
        _landmarks = np.load(self.landmarks_path, allow_pickle=True)

        tiff_qimage = QImage(self.tiff_image.convert("RGB").tobytes(), self.tiff_image.width, self.tiff_image.height, QImage.Format_RGB888)
        self.tiff_label.setPixmap(QPixmap.fromImage(tiff_qimage))

        self.points_image1 = _landmarks[0]
        self.points_image2 = _landmarks[1]

        # print(self.points_image1, self.points_image2)
        _, ext = os.path.splitext(self.czi_path)
        ext = ext.lower()
        if ext == ".czi":
            self.czi = CziFile(self.czi_path)
            self.mosaic_data = self.czi.read_mosaic(C=0, scale_factor=1)
            self.czi_array = self.mosaic_data.squeeze()

        elif ext == ".tif" or ext == ".tiff":
            with TiffFile(self.czi_path) as tif:
                if not tif.series:
                    raise ValueError("No image series found in TIFF file.")
        
                # Access the first series
                first_series = tif.series[0]
                self.czi_array = first_series.asarray()
                print(self.czi_array.shape)

        else:
            return

        png_y = self.png_image.size[1]
        czi_y = self.czi_array.shape[0]
        
        self.scale_ratio = czi_y // png_y


        _png_img = np.array(self.png_image.convert("RGB"))
        _text = '1/'+str(self.scale_ratio)+' X'
        print(int(_png_img.shape[0] - _png_img.shape[0]/20), int(_png_img.shape[1]/20))
        _png_img = cv2.putText(_png_img, _text, (int(_png_img.shape[0] - _png_img.shape[0]/20), int(200)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)

        png_qimage =QImage(_png_img.data, _png_img.shape[1], _png_img.shape[0], _png_img.strides[0], QImage.Format_RGB888)
        self.png_label.setPixmap(QPixmap.fromImage(png_qimage))
      

        self.status_label.setText("Ready! Click on the image to view the region in the CZI.")

    def on_click(self, event):
        if self.czi_array is None:
            return

        x_tiff = event.pos().x()
        y_tiff = event.pos().y()

        x_png, y_png = self.thin_plate(x_tiff, y_tiff)
        print(x_tiff, y_tiff)
        print(x_png, y_png)

        x_czi = int(x_png * self.scale_ratio)
        y_czi = int(y_png * self.scale_ratio)

        region_size = 1024
        x_start = max(x_czi - region_size // 2, 0)
        y_start = max(y_czi - region_size // 2, 0)
        x_end = min(x_start + region_size, self.czi_array.shape[1])
        y_end = min(y_start + region_size, self.czi_array.shape[0])
        x_start = max(0, x_end - region_size)
        y_start = max(0, y_end - region_size)

        region = self.czi_array[y_start:y_end, x_start:x_end, :]

        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB) if region.shape[2] == 3 else region
        region_qimage = QImage(region_rgb.data, region_rgb.shape[1], region_rgb.shape[0], region_rgb.strides[0], QImage.Format_RGB888)
        self.region_label.setPixmap(QPixmap.fromImage(region_qimage))

        # Draw red rectangle on PNG preview
        png_copy = self.png_image.convert("RGB").copy()
        
        x_rect = int(x_png - region_size/self.scale_ratio/2)
        y_rect = int(y_png - region_size/self.scale_ratio/2)
        w_rect = h_rect = int(region_size/self.scale_ratio)

        cv_img = np.array(png_copy)
        cv_img = cv2.rectangle(cv_img, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (255, 0, 0), 3)

        updated_png_qimage = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format_RGB888)
        self.png_label.setPixmap(QPixmap.fromImage(updated_png_qimage))


        # if self.last_figure:
        #     plt.close(self.last_figure)

        # self.last_figure = plt.figure(figsize=(15, 15))
        # plt.imshow(region, cmap='gray' if region.shape[-1] == 1 else None)
        # plt.title(f"CZI Region at ({x_czi}, {y_czi})")
        # plt.axis('off')
        # plt.show()


    def thin_plate(self, x_tiff, y_tiff):
        if len(self.points_image1) == len(self.points_image2):
        
            pts_img1 = np.array([[point.x(), point.y()] for point in self.points_image1]) 
            pts_img2 = np.array([[point.x(), point.y()] for point in self.points_image2])
       
            pts_img1=pts_img1.reshape(-1,len(pts_img1),2)
            pts_img2=pts_img2.reshape(-1,len(pts_img2),2)
        
            # img1 = cv2.imread(self.tiff_path)
            # img2 = cv2.imread(self.png_path)


            splines= cv2.createThinPlateSplineShapeTransformer()
            matches=list()
            for i in range(0,len(pts_img1[0])):

                matches.append(cv2.DMatch(i,i,0))

            temp=splines.estimateTransformation(pts_img1,pts_img2,matches)

            f32_pts = np.array([[[x_tiff,y_tiff]]], dtype=np.float32)
            transformed_point = splines.applyTransformation(f32_pts)
            # transformed_point =  transform_func([x, y])
            tx, ty = transformed_point[1][0][0]

            return tx, ty

        else:
            print("\033[31mNumber of points selected on both images is not equal!\033[0m")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = CZIMatcherApp()
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec_())
