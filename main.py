from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog
from ui import Ui_MainWindow  # importing our generated file
import sys
from os import listdir
from os.path import isfile, join
from model_api import model_loader,image_reader,image_predict

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.img = ''
        self.mask = ''
        self.model = ''

        self.ui = Ui_MainWindow()   
        self.ui.setupUi(self)
        self.ui.s_image_label.clicked.connect(self.select_image)
        self.ui.s_model.clicked.connect(self.select_model)
        self.ui.run.clicked.connect(self.execute)

    def select_image(self):
        image_path, filetype = QFileDialog.getOpenFileName(self,"選取原始圖片","./","All Files (*);;Text Files (*.txt)")
        label_path, filetype = QFileDialog.getOpenFileName(self,"原始圖片對應的mask","./","All Files (*);;Text Files (*.txt)")
        self.ui.image_path.setText(image_path)
        self.ui.label_path.setText(label_path)
        self.img = image_reader(image_path)
        self.mask = image_reader(label_path)

        #return image_path,label_path
    def select_model(self):
        model_path, filetype = QFileDialog.getOpenFileName(self,"選取模型","./","All Files (*);;Text Files (*.txt)")
        #print(model_path)
        self.ui.model_name.setText(model_path)
        self.model = model_loader(model_path)
        #return model_path

    def execute(self):
        dc_list = image_predict(self.img,self.mask,self.model)
        self.ui.picture.setPixmap(QtGui.QPixmap("final_result.png"))

        self.ui.label.setText('DC 1: '+ str(dc_list[0]))
        self.ui.label_2.setText('DC 2: '+str(dc_list[1]))
        self.ui.label_3.setText('DC 3: '+str(dc_list[2]))
        self.ui.label_4.setText('DC 4: '+str(dc_list[3]))
        self.ui.label_5.setText('DC 5: '+str(dc_list[4]))
        self.ui.label_6.setText('DC 6: '+str(dc_list[5]))
        self.ui.label_7.setText('DC 7: '+str(dc_list[6]))
        self.ui.label_8.setText('DC 8: '+str(dc_list[7]))
        self.ui.label_9.setText('DC 9: '+str(dc_list[8]))
        self.ui.label_10.setText('DC 10: '+str(dc_list[9]))
        self.ui.label_11.setText('DC 11: '+str(dc_list[10]))
        self.ui.label_12.setText('DC 12: '+str(dc_list[11]))
        self.ui.label_13.setText('DC 13: '+str(dc_list[12]))
        self.ui.label_14.setText('DC 14: '+str(dc_list[13]))
        self.ui.label_15.setText('DC 15: '+str(dc_list[14]))
        self.ui.label_16.setText('DC 16: '+str(dc_list[15]))

        self.ui.label_17.setText('DC 17: '+str(dc_list[16]))
        self.ui.label_18.setText('DC 18: '+str(dc_list[17]))
        self.ui.label_19.setText('DC 19: '+str(dc_list[18]))
        self.ui.label_20.setText('DC AVG: '+ str(dc_list[19]))


app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())