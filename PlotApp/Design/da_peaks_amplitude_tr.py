# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'da_peaks_amplitude.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(412, 261)
        font = QtGui.QFont()
        font.setPointSize(7)
        Dialog.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_da = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_da.setFont(font)
        self.label_da.setObjectName("label_da")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_da)
        self.doubleSpinBox_da = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_da.setMaximumSize(QtCore.QSize(70, 16777215))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.doubleSpinBox_da.setFont(font)
        self.doubleSpinBox_da.setMinimum(-99.99)
        self.doubleSpinBox_da.setSingleStep(0.1)
        self.doubleSpinBox_da.setObjectName("doubleSpinBox_da")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_da)
        self.pushButtonCalculate_da = QtWidgets.QPushButton(Dialog)
        self.pushButtonCalculate_da.setObjectName("pushButtonCalculate_da")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButtonCalculate_da)
        self.pushButtonPlot_da = QtWidgets.QPushButton(Dialog)
        self.pushButtonPlot_da.setObjectName("pushButtonPlot_da")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.pushButtonPlot_da)
        self.labelPeaksNumber_da = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.labelPeaksNumber_da.setFont(font)
        self.labelPeaksNumber_da.setText("")
        self.labelPeaksNumber_da.setObjectName("labelPeaksNumber_da")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.labelPeaksNumber_da)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "DA_amplitude"))
        self.label_da.setText(_translate("Dialog", "Amplitude condition:"))
        self.pushButtonCalculate_da.setText(_translate("Dialog", "Calculate"))
        self.pushButtonPlot_da.setText(_translate("Dialog", "Plot AC\'s"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
