import sys
import os.path
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from PIL import ImageOps
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog

#Gerekli kütüphaneleri ve paketleri yüklemeyi unutmayın
#Ömer Burak Özgür
#194410013
#omerburakozgur1@gmail.com +90507 027 1482
#Kastamonu Üniversitesi

#========================================================

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32
imageLocation = ""
outputGenerated = False
imageName = ""

#========================================================

def segmenteEt():
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))

    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)

    # ========================================================

    class OxfordPets(keras.utils.Sequence):
        """Helper to iterate over the data (as Numpy arrays)."""

        def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
            self.batch_size = batch_size
            self.img_size = img_size
            self.input_img_paths = input_img_paths
            self.target_img_paths = target_img_paths

        def __len__(self):
            return len(self.target_img_paths) // self.batch_size

        def __getitem__(self, idx):
            """Returns tuple (input, target) correspond to batch #idx."""
            i = idx * self.batch_size
            batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
            batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
            x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
            for j, path in enumerate(batch_input_img_paths):
                img = load_img(path, target_size=self.img_size)
                x[j] = img
            y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                y[j] = np.expand_dims(img, 2)
                # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
                y[j] -= 1
            return x, y

    # ========================================================

    val_samples = 100
    # random.Random(1337).shuffle(input_img_paths)
    # random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    model = tf.keras.models.load_model("oxfordpet.h5")

    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_preds = model.predict(val_gen)
    # Display results for validation image #10
    i = 0
    # Display ground-truth target mask
    img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    img = PIL.ImageOps.fit(img, img_size)
    #plt.imshow(img)
    #plt.show()
    plt.imshow(val_preds[i])
    #plt.axis("off")
    plt.savefig("output.jpg")
    outputGenerated = True

    plt.show()

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        #Load UI
        uic.loadUi("test.ui",self)

        #Define Widgets
        self.btnSec = self.findChild(QPushButton, "btnGoruntuSec")
        self.btnKaydet = self.findChild(QPushButton, "btnGoruntuKaydet")
        self.lblCikis = self.findChild(QLabel, "lblGirisGoruntusu")
        self.lblGiris = self.findChild(QLabel, "lblCikisGoruntusu")

        #Dropdown Box
        self.btnSec.clicked.connect(self.goruntuSec)
        self.btnKaydet.clicked.connect(self.goruntuKaydet)

        #Show the App
        self.show()

    def goruntuSec(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "D:\\", "All Files (*)")
        imageName = os.path.basename(fname[0])
        if len(fname[0])>=2:
            #Open The Image
            self.pixmap = QPixmap(fname[0])
            #Add pic to label
            self.lblCikis.setPixmap(self.pixmap)
            # Python dosyası üzerinden çalıştırıyorsanız
            # aşşağıdaki dizini projeyi attığınız konuma göre değiştirmeniz gerekmektedir.
            shutil.copy(fname[0], "D:\\Dosyalar\\Ders\\PythonGUI\\images")
            if os.path.exists("output.jpg"):
                os.remove("output.jpg")
            segmenteEt()
            if os.path.exists("output.jpg"):
                #Python dosyası üzerinden çalıştırıyorsanız
                #aşşağıdaki dizini projeyi attığınız konuma göre değiştirmeniz gerekmektedir.
                self.pixmap2 = QPixmap("D:\\Dosyalar\\Ders\\PythonGUI\\output.jpg")
                self.lblGiris.setPixmap(self.pixmap2)
                os.remove("D:\\Dosyalar\\Ders\\PythonGUI\\images\\"+imageName)
                os.remove("output.jpg")
        else:
            print("Dialog Exit")


    def goruntuKaydet(self):
        saveLocation = QFileDialog.getExistingDirectory(self, "Select Save Location", "c:\\")
        # Python dosyası üzerinden çalıştırıyorsanız
        # aşşağıdaki dizini projeyi attığınız konuma göre değiştirmeniz gerekmektedir.
        shutil.copy("D:\\Dosyalar\\Ders\\PythonGUI\\output.jpg", saveLocation)
        print("Save Location and File Name: "+ saveLocation)





app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()