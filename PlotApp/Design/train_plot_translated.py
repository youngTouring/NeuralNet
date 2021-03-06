# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'train_plot.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1234, 608)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.widgetPlot = QtWidgets.QWidget(self.centralwidget)
        self.widgetPlot.setMaximumSize(QtCore.QSize(16777215, 1))
        self.widgetPlot.setObjectName("widgetPlot")
        self.gridLayout.addWidget(self.widgetPlot, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1234, 26))
        self.menubar.setObjectName("menubar")
        self.menuPlik = QtWidgets.QMenu(self.menubar)
        self.menuPlik.setObjectName("menuPlik")
        self.menuAction = QtWidgets.QMenu(self.menubar)
        self.menuAction.setObjectName("menuAction")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOtw_rz_plik = QtWidgets.QAction(MainWindow)
        self.actionOtw_rz_plik.setObjectName("actionOtw_rz_plik")
        self.actionZapisz_plik = QtWidgets.QAction(MainWindow)
        self.actionZapisz_plik.setObjectName("actionZapisz_plik")
        self.actionDA_amplitude = QtWidgets.QAction(MainWindow)
        self.actionDA_amplitude.setObjectName("actionDA_amplitude")
        self.actionNON_DA_amplitude = QtWidgets.QAction(MainWindow)
        self.actionNON_DA_amplitude.setObjectName("actionNON_DA_amplitude")
        self.menuPlik.addAction(self.actionOtw_rz_plik)
        self.menuPlik.addAction(self.actionZapisz_plik)
        self.menuAction.addAction(self.actionDA_amplitude)
        self.menuAction.addAction(self.actionNON_DA_amplitude)
        self.menubar.addAction(self.menuPlik.menuAction())
        self.menubar.addAction(self.menuAction.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "train_plot"))
        self.menuPlik.setTitle(_translate("MainWindow", "File"))
        self.menuAction.setTitle(_translate("MainWindow", "Action"))
        self.actionOtw_rz_plik.setText(_translate("MainWindow", "Open"))
        self.actionOtw_rz_plik.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionZapisz_plik.setText(_translate("MainWindow", "Save"))
        self.actionZapisz_plik.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionDA_amplitude.setText(_translate("MainWindow", "DA amplitude"))
        self.actionNON_DA_amplitude.setText(_translate("MainWindow", "NON_DA amplitude"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
