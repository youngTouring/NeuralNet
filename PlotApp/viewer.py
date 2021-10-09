from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from scipy.signal import savgol_filter, peak_widths, peak_prominences, find_peaks
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PlotApp.Design.viewer_tr import *
from matplotlib.figure import Figure
import tensorflow as tf
import pandas as pd
import numpy as np
import traceback
import pickle
import h5py
import sys
import os

class MplCanvas(FigureCanvas):
    """Plotting class"""
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        fig = Figure(figsize=(width, height), dpi=dpi,tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Viewer(QMainWindow):
    """Main window class"""
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOpen.triggered.connect(self.OpenSignal)

    def OpenSignal(self):
        while self.ui.gridLayout.count():
            child = self.ui.gridLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.data = None
        self.peaks = None
        self.prominence = None
        self.width = None
        self.minima = None
        self.labels = None
        self.plot = None
        self.signal_dataframe = pd.DataFrame
        file, _ = QFileDialog.getOpenFileName(self, "Open files", "","All files (*);;H5 files (*.h5)")
        if file:
            try:
                data = h5py.File(file, 'r')
                self.data = data.get('signal')
                self.data = np.array(self.data)
                self.data = savgol_filter(self.data, 75, 6)
                self.sc = MplCanvas(self, width=5, height=2, dpi=120)
                self.sc.setFocusPolicy(QtCore.Qt.ClickFocus)
                self.sc.setFocus()
                self.sc.axes.plot(self.data)
                self.sc.axes.grid()
                toolbar = NavigationToolbar(self.sc, self)
                self.ui.gridLayout.addWidget(toolbar)
                self.ui.gridLayout.addWidget(self.sc)
                self.xdataPlot = np.arange(0, len(self.data), dtype=float)
                self.ydataPlot = self.data
                file_name = str(file).split('/')
                self.file_name_no_extension = file_name[-1].split('.')
                self.sc.axes.set_title(self.file_name_no_extension[0])
                self.AddPeaks()
                self.PlotPeaks()
            except Exception as e:
                '''In case of incorrect file extension show error'''
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                QMessageBox.critical(self, 'Error', f'Something went wrong: \n{e}')

    def AddPeaks(self):
        try:
            peaks_height = 0.3
            self.peaks, peaks_propertes = find_peaks(self.data, height=peaks_height,distance=150)
            minima_points_y = []
            minima_points_x = []
            for i,j in zip(self.peaks,self.peaks + 100):
                point_y = np.min(self.data[i:j])
                point_x = np.where(self.data == point_y)
                minima_points_x.append(point_x[0].item(0))
                minima_points_y.append(point_y)
            minima_y = np.array(minima_points_y)
            minima_x = np.array(minima_points_x)
            self.peaks = np.vstack((self.peaks,self.data[self.peaks])).T
            self.minima = np.vstack((minima_x,minima_y)).T
            peaks_x = self.peaks[:, 0].astype(np.int)
            self.prominence, prominence = peak_prominences(self.data, peaks_x)[0], peak_prominences(self.data, peaks_x)
            self.width = peak_widths(self.data, peaks_x, prominence_data=prominence,
                                                  rel_height=0.5)
            self.width = np.clip(self.width,0,65)
        except Exception as e:
            QMessageBox.critical(self,'Info',e)

    def PlotPeaks(self):
        try:
            data_dict = {'Peaks': self.peaks[:,1], 'Amplitude': self.prominence[0],
                              'Width': self.width[0],'Minima': self.minima[:,1]}
            self.signal_dataframe = pd.DataFrame(data=data_dict)
            data = self.signal_dataframe
            scaler = StandardScaler()
            train_data,test_data = train_test_split(data,test_size=0.2,shuffle=False)
            scaler = StandardScaler().fit(train_data)
            data = scaler.transform(data)
            classifier = tf.keras.models.load_model('peaks_classifier_model.h5')
            prediction = classifier.predict_classes(data)
            data = scaler.inverse_transform(data)
            self.signal_dataframe['Labels'] = prediction
            non_da = self.signal_dataframe.loc[self.signal_dataframe['Labels'] == 0,'Peaks']
            da = self.signal_dataframe.loc[self.signal_dataframe['Labels'] == 1,'Peaks']
            non_da = non_da.values
            da = da.values
            non_da = np.array(non_da)
            da = np.array(da)
            x_non_da = []
            x_da = []
            for i in non_da:
                x_non_da_point = np.where(self.data == i)
                x_non_da.append(x_non_da_point[0].item(0))
            for i in da:
                x_da_point = np.where(self.data == i)
                x_da.append(x_da_point[0].item(0))
            self.sc.axes.scatter(x_non_da,non_da,color='black')
            self.sc.axes.scatter(x_da,da, color='red')
        except Exception as e:
            QMessageBox.critical(self,'Error',e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())