from PyQt5 import QtWidgets, QtGui, QtCore, Qt
from Params import Params


class WidgetParams(QtWidgets.QWidget):
    def __init__(self, parent=None, x=0, y=0):
        self.params = Params()

        QtWidgets.QWidget.__init__(self, parent)
        self.setGeometry(QtCore.QRect(x, y, 450, 600))
        self.setObjectName("widget_params")
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(20, 100, 371, 321))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(80, 10, 300, 25))
        self.label.setObjectName("label")
        self.checkBox = QtWidgets.QCheckBox(self.widget)
        self.checkBox.setGeometry(QtCore.QRect(16, 35, 201, 20))
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setGeometry(QtCore.QRect(10, 50, 351, 191))
        self.widget_2.setObjectName("widget_2")
        self.widget1 = QtWidgets.QWidget(self.widget_2)
        self.widget1.setGeometry(QtCore.QRect(10, 10, 341, 176))
        self.widget1.setObjectName("widget1")
        self.formLayout = QtWidgets.QFormLayout(self.widget1)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_3 = QtWidgets.QLabel(self.widget1)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.label_4 = QtWidgets.QLabel(self.widget1)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_5 = QtWidgets.QLabel(self.widget1)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_6 = QtWidgets.QLabel(self.widget1)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.label_7 = QtWidgets.QLabel(self.widget1)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_6)
        self.widget_3 = QtWidgets.QWidget(self)
        self.widget_3.setGeometry(QtCore.QRect(20, 340, 371, 150))
        self.widget_3.setObjectName("widget_3")
        self.label_8 = QtWidgets.QLabel(self.widget_3)
        self.label_8.setGeometry(QtCore.QRect(80, 10, 450, 25))
        self.label_8.setObjectName("label_8")
        self.checkBox_2 = QtWidgets.QCheckBox(self.widget_3)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 35, 300, 25))
        self.checkBox_2.setTristate(False)
        self.checkBox_2.setObjectName("checkBox_2")
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setGeometry(QtCore.QRect(21, 61, 150, 20))
        self.label_9.setObjectName("label_9")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_7.setGeometry(QtCore.QRect(240, 60, 114, 19))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.widget_4 = QtWidgets.QWidget(self)
        self.widget_4.setGeometry(QtCore.QRect(20, 10, 371, 101))
        self.widget_4.setObjectName("widget_4")
        self.label_10 = QtWidgets.QLabel(self.widget_4)
        self.label_10.setGeometry(QtCore.QRect(130, 10, 171, 20))
        self.label_10.setObjectName("label_10")
        self.widget_41 = QtWidgets.QWidget(self.widget_4)
        self.widget_41.setGeometry(QtCore.QRect(100, 40, 201, 46))
        self.widget_41.setObjectName("widget_41")
        self.formLayout_2 = QtWidgets.QFormLayout(self.widget_41)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_11 = QtWidgets.QLabel(self.widget_41)
        self.label_11.setObjectName("label_11")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.widget_41)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_8)
        self.label_12 = QtWidgets.QLabel(self.widget_41)
        self.label_12.setObjectName("label_12")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.widget_41)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_9)

        self.label.setText("<b>Параметры поиска таблицы</b>")
        self.checkBox.setText("Эмулировать таблицу")
        self.label_2.setText("Количество размытий")
        self.label_3.setText("Величина порога")
        self.label_4.setText("Длина горизонтальной линии")
        self.label_5.setText("Минимальное расстояние между линиями")
        self.label_6.setText("Ширина коридора")
        self.label_7.setText("Количество точек в линии")
        self.label_10.setText("Обрезать")
        self.label_8.setText("<b>Параметры картинки с текстом</b>")
        self.checkBox_2.setText("Чистить границы")
        self.label_9.setText("Величина порога")
        self.label_10.setText("<b>Предобработка</b>")
        self.label_11.setText("Поворот на")
        self.label_12.setText("Обрезать края")

        self.lineEdit.setText(str(self.params.count_of_blur))
        self.lineEdit_2.setText(str(self.params.thr_border))
        self.lineEdit_3.setText(str(self.params.hough_param_gor))
        self.lineEdit_4.setText(str(self.params.min_dist_between_points))
        self.lineEdit_5.setText(str(self.params.coridor))
        self.lineEdit_6.setText(str(self.params.count_of_points))
        self.lineEdit_8.setText(str(self.params.cut_width))
        self.lineEdit_9.setText(str(self.params.angle))
        self.lineEdit_7.setText(str(self.params.thr_text))

        self.checkBox.clicked.connect(self.clickedEmulation)
        self.lineEdit.textChanged.connect(self.setCountBlur)
        self.lineEdit_2.textChanged.connect(self.setThrTable)
        self.lineEdit_3.textChanged.connect(self.setHoughGor)
        self.lineEdit_4.textChanged.connect(self.setMinDistBetweenPoints)
        self.lineEdit_5.textChanged.connect(self.setCoridor)
        self.lineEdit_6.textChanged.connect(self.setCountOfPoints)
        self.lineEdit_8.textChanged.connect(self.setAngle)
        self.lineEdit_9.textChanged.connect(self.setCutWidth)

        self.checkBox_2.clicked.connect(self.clickedKillFrame)
        self.lineEdit_7.textChanged.connect(self.setThresh)

    def clickedEmulation(self):
        self.widget1.setVisible(not self.checkBox.isChecked())
        self.params.emulate_border = self.checkBox.isChecked()

    def setCountBlur(self):
        if self.lineEdit.text().isdigit():
            self.params.count_of_blur = int(self.lineEdit.text())

    def setThrTable(self):
        if self.lineEdit_2.text().isdigit():
            self.params.thr_border = int(self.lineEdit_2.text())

    def setHoughGor(self):
        if self.lineEdit_3.text().isdigit():
            self.params.hough_param_gor = int(self.lineEdit_3.text())

    def setMinDistBetweenPoints(self):
        if self.lineEdit_4.text().isdigit():
            self.params.min_dist_between_points = int(self.lineEdit_4.text())

    def setCoridor(self):
        if self.lineEdit_5.text().isdigit():
            self.params.coridor = int(self.lineEdit_5.text())

    def setCountOfPoints(self):
        if self.lineEdit_6.text().isdigit():
            self.params.count_of_points = int(self.lineEdit_6.text())

    def setCutWidth(self):
        if self.lineEdit_9.text().isdigit():
            self.params.cut_width = int(self.lineEdit_9.text())

    def setAngle(self):
        if self.lineEdit_8.text().isdigit():
            self.params.angle = int(self.lineEdit_8.text())

    def clickedKillFrame(self):
        self.params.kill_border = self.checkBox_2.isChecked()

    def setThresh(self):
        if self.lineEdit_7.text().isdigit():
            self.params.thr_text = int(self.lineEdit_7.text())


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = WidgetParams()
    # w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())
