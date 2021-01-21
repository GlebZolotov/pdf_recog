from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from mainwindow import MainWindow
import sys
import os
from ModelApp import Model
import logging
import PageWindow


class MainWindowUIClass(MainWindow):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.widget1.hide()
        self.widget2.hide()
        self.pageWindow = PageWindow.PageWindow(mainW=self)

    def refreshAll(self):
        self.lineEdit.setText(self.model.getFileName())
        self.lineEdit2.setText(self.model.getFileName())

    # slot
    def returnPressedSlot(self):
        fileName = self.lineEdit.text()
        if self.model.isValid(fileName):
            self.model.setFileName(self.lineEdit.text())
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + fileName)
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit.setText("")
            self.refreshAll()

    # slot
    def returnPressedImgSlot(self):
        fileName = self.lineEdit2.text()
        if self.model.isValid(fileName):
            self.model.setFileName(self.lineEdit2.text())
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + fileName)
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit2.setText("")
            self.refreshAll()

    # slot
    def browseSlot(self):
        # self.debugPrint("Browse button pressed")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;PDF (*.pdf)",
            options=options)
        if fileName:
            self.model.setFileNamePDF(fileName)
            self.refreshAll()

    # slot
    def browseImgSlot(self):
        # self.debugPrint("Browse button pressed")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;JPEG (*.jpg)",
            options=options)
        if fileName:
            self.model.setFileName(fileName)
            self.refreshAll()

    # slot
    def seeWidgetSlot(self):
        self.widget1.setVisible(not self.widget1.isVisible())
        if self.widget2.isVisible() and self.widget1.isVisible():
            self.widget2.setVisible(False)

    # slot
    def seeWidgetImgSlot(self):
        self.widget2.setVisible(not self.widget2.isVisible())
        if self.widget2.isVisible() and self.widget1.isVisible():
            self.widget1.setVisible(False)

    # slot
    def UniteCsvSlot(self):
        if os.path.exists(self.model.foldName + ".csv"):
            os.remove(self.model.foldName + ".csv")
        with open(self.model.foldName + ".csv", "a") as fout:
            for num in range(self.model.count_of_pages):
                fname = os.path.join(self.model.foldName, "Source", str(num + 1) + ".csv")
                print(fname)
                if not os.path.exists(fname):
                    continue
                with open(fname, "r") as f1:
                    for line in f1:
                        fout.write(line)

    # slot
    def startRecSlot(self):
        if self.model.getFileName() != "":
            name_imgs = self.model.load_file()
            # self.model.recog_imgs(name_imgs, self.widgetParams.getHoughGor(), self.widgetParams.getCountOfPoints(), self.widgetParams.getCoridor(), self.widgetParams.getMinDistBetweenPoints(), self.widgetParams.getThresh())

    # slot
    def startImgSlot(self):
        self.pageWindow.setImage(self.model.getFileName())
        self.pageWindow.showMaximized()
        self.close()


def create_logger():
    logger = logging.getLogger("startApp")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler("log.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    logger.info("Logging started")


def main():
    create_logger()
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindowUIClass()
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
