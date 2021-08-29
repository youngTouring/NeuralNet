import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PlotApp.Design.train_plot_translated import *
from da_peaks_amplitude import *
from non_da_peaks_amplitude import *
import tensorflow as tf
from scipy.signal import savgol_filter, peak_widths, peak_prominences
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

class TrainingPlot(QMainWindow):
    """Main window class"""
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOtw_rz_plik.triggered.connect(self.OpenFile)
        self.ui.actionZapisz_plik.triggered.connect(self.SaveDataset)
        self.file_name_no_extension = ''

        self.da_added = np.empty(shape=2)
        self.da_minima_points_added = np.empty(shape=2)
        self.da_plot_added = None
        self.da_minima_plot_added = None
        self.prominence_da = None
        self.prominence_properties_da = None
        self.peaks_width_da = None

        self.non_da_added = np.empty(shape=2)
        self.non_da_minima_points_added = np.empty(shape=2)
        self.non_da_plot_added = None
        self.minima_non_da_plot = None
        self.prominence_non_da = None
        self.prominence_properties_non_da = None
        self.peaks_width_non_da = None

        self.ui.actionNON_DA_amplitude.triggered.connect(self.OpenNonDaAmplitude)
        self.ui.actionDA_amplitude.triggered.connect(self.OpenDaAmplitude)

        self.da_amplitude_dialog = DaPeaks()
        self.da_amplitude_dialog.ui.pushButtonPlot_da.clicked.connect(self.AddDaPeaks)

        self.non_da_amplitude_dialog = NonDaPeaks()
        self.non_da_amplitude_dialog.ui.pushButtonPlot_non_da.clicked.connect(self.AddNonDaPeaks)

    def OpenFile(self):
        """Reads data from file and sets up signal plot, with scipy labels"""
        while self.ui.gridLayout.count():
            child = self.ui.gridLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.file_name_no_extension = ''
        self.non_da_added = np.empty(shape=2)
        self.non_da_minima_points_added = np.empty(shape=2)
        self.non_da_plot_added = None
        self.minima_non_da_plot = None
        self.prominence_non_da = None
        self.prominence_properties_non_da = None
        self.peaks_width_non_da = None

        self.da_added = np.empty(shape=2)
        self.da_minima_points_added = np.empty(shape=2)
        self.da_plot_added = None
        self.da_minima_plot_added = None
        self.prominence_da = None
        self.prominence_properties_da = None
        self.peaks_width_da = None

        self.new_peaks_width = None
        self.new_peaks_width_left = None

        self.minima_da_plot = None
        self.minima_da_plot_added = None
        self.minima_non_da_plot = None
        self.minima_non_da_plot_added = None
        self.da_plot = None
        self.da_plot_added = None
        self.non_da_plot = None
        self.non_da_plot_added = None

        file, _ = QFileDialog.getOpenFileName(self,"Open files","",
                                              "All files (*);;H5 files (*.h5)")
        if file:
            try:
                ### Plot and data
                data = h5py.File(file, 'r')
                self.data = data.get('signal')
                self.data = np.array(self.data)
                self.data = savgol_filter(self.data, 75, 6)
                self.sc = MplCanvas(self, width=5, height=2, dpi=120)
                self.sc.setFocusPolicy(QtCore.Qt.ClickFocus)
                self.sc.setFocus()
                self.sc.mpl_connect('key_press_event', self.ChangePeaksClass)
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
            except Exception as e:
                '''In case of incorrect file extension show error'''
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                QMessageBox.critical(self, 'Error', f'Something went wrong: \n{e}')

    def OpenDaAmplitude(self):
        try:
            self.da_amplitude_dialog.data = self.data
            self.da_amplitude_dialog.show()
        except Exception as e:
            QMessageBox.information(self,'Inormation','No data for amplitude condition.\n'
                                                      'Please load signal first')

    def OpenNonDaAmplitude(self):
        try:
            self.non_da_amplitude_dialog.data = self.data
            self.non_da_amplitude_dialog.show()
        except Exception as e:
            QMessageBox.information(self, 'Inormation', 'No data for amplitude condition.\n'
                                                        'Please load signal first')

    def AddDaPeaks(self):
        try:
            self.da_plot = self.sc.axes.scatter(self.da_amplitude_dialog.da_peaks[:,0],self.da_amplitude_dialog.da_peaks[:,1],color='red')
            self.minima_da_plot = self.sc.axes.scatter(self.da_amplitude_dialog.minima[:,0],self.da_amplitude_dialog.minima[:,1],color='green')
            self.da_amplitude_dialog.close()
        except Exception as e:
            QMessageBox.critical(self,'Error',e)

    def AddNonDaPeaks(self):
        try:
            self.non_da_plot = self.sc.axes.scatter(self.non_da_amplitude_dialog.non_da_peaks[:,0],self.non_da_amplitude_dialog.non_da_peaks[:,1],
                                                    color='black')
            self.minima_non_da_plot = self.sc.axes.scatter(self.non_da_amplitude_dialog.minima[:,0],self.non_da_amplitude_dialog.minima[:,1],
                                                           color='blue')
            self.non_da_amplitude_dialog.close()
        except Exception as e:
            QMessageBox.critical(self,'Error',e)

    def ChangePeaksClass(self,event):
        """Deletes labels from self.non_da and self.da. Deleting label from plot - in progress"""
        if event.key == 'z':
            xdata_click = event.xdata
            xdata_nearest = (np.abs(self.xdataPlot - xdata_click)).argmin()
            deletion_range = self.xdataPlot[(self.xdataPlot >= (xdata_nearest - 350)) * (self.xdataPlot <= (xdata_nearest + 350))]
            for i in deletion_range:
                if self.non_da_plot != None:
                    if i in self.non_da_amplitude_dialog.non_da_peaks[:, 0]:
                        value_to_delete = np.where(self.non_da_amplitude_dialog.non_da_peaks == i)
                        value_to_delete_minima = value_to_delete[0].item(0)
                        value_to_add = value_to_delete[0].item(0)
                        point_to_add = self.non_da_amplitude_dialog.non_da_peaks[value_to_add]
                        self.da_added = np.insert(self.da_added, 0, point_to_add, axis=0)
                        self.da_added = np.reshape(self.da_added, (-1, 2))
                        self.non_da_amplitude_dialog.non_da_peaks = np.delete(self.non_da_amplitude_dialog.non_da_peaks,
                                                                              np.where(self.non_da_amplitude_dialog.non_da_peaks[:, 0] == i)
                                                                              ,axis=0)
                        self.non_da_amplitude_dialog.non_da_peaks = np.reshape(self.non_da_amplitude_dialog.non_da_peaks, (-1, 2))
                        self.non_da_plot.set_offsets(self.non_da_amplitude_dialog.non_da_peaks)
                        minima_point_to_add = self.non_da_amplitude_dialog.minima[value_to_add]
                        self.da_minima_points_added = np.insert(self.da_minima_points_added, 0, minima_point_to_add, axis=0)
                        self.da_minima_points_added = np.reshape(self.da_minima_points_added, (-1, 2))
                        self.non_da_amplitude_dialog.minima = np.delete(self.non_da_amplitude_dialog.minima,value_to_delete_minima, axis=0)
                        self.non_da_amplitude_dialog.minima = np.reshape(self.non_da_amplitude_dialog.minima, (-1, 2))
                        self.minima_non_da_plot.set_offsets(self.non_da_amplitude_dialog.minima)
                        self.Update_da_plot()
                        break
                if self.da_plot != None:
                    if i in self.da_amplitude_dialog.da_peaks[:,0]:
                        value_to_delete = np.where(self.da_amplitude_dialog.da_peaks == i)
                        value_to_delete_minima = value_to_delete[0].item(0)
                        value_to_add = value_to_delete[0].item(0)
                        point_to_add = self.da_amplitude_dialog.da_peaks[value_to_add]
                        self.non_da_added = np.insert(self.non_da_added, 0, point_to_add, axis=0)
                        self.non_da_added = np.reshape(self.non_da_added, (-1, 2))
                        self.da_amplitude_dialog.da_peaks = np.delete(self.da_amplitude_dialog.da_peaks,
                                                                              np.where(self.da_amplitude_dialog.da_peaks[:, 0] == i)
                                                                              ,axis=0)
                        self.da_amplitude_dialog.da_peaks = np.reshape(self.da_amplitude_dialog.da_peaks, (-1, 2))
                        self.da_plot.set_offsets(self.da_amplitude_dialog.da_peaks)
                        minima_point_to_add = self.da_amplitude_dialog.minima[value_to_add]
                        self.non_da_minima_points_added = np.insert(self.non_da_minima_points_added, 0, minima_point_to_add,axis=0)
                        self.non_da_minima_points_added = np.reshape(self.non_da_minima_points_added, (-1, 2))
                        self.da_amplitude_dialog.minima = np.delete(self.da_amplitude_dialog.minima,value_to_delete_minima, axis=0)
                        self.da_amplitude_dialog.minima = np.reshape(self.da_amplitude_dialog.minima, (-1, 2))
                        self.minima_da_plot.set_offsets(self.da_amplitude_dialog.minima)
                        self.Update_non_da_plot()
                        break

    def Update_non_da_plot(self):
        non_da = self.non_da_added[:-1]
        minima = self.non_da_minima_points_added[:-1]
        if self.non_da_plot_added == None:
            self.non_da_plot_added = self.sc.axes.scatter(non_da[:, 0], non_da[:, 1], color='black')
            self.minima_non_da_plot_added = self.sc.axes.scatter(minima[:,0],minima[:,1],color='blue')
        else:
            self.non_da_plot_added.set_offsets(non_da)
            self.minima_non_da_plot_added.set_offsets(minima)
            self.sc.draw()

    def Update_da_plot(self):
        da = self.da_added[:-1]
        minima = self.da_minima_points_added[:-1]
        if self.da_plot_added == None:
            self.da_plot_added = self.sc.axes.scatter(da[:, 0], da[:, 1], color='red')
            self.minima_da_plot_added = self.sc.axes.scatter(minima[:, 0], minima[:, 1], color='green')
        else:
            self.da_plot_added.set_offsets(da)
            self.minima_da_plot_added.set_offsets(minima)
            self.sc.draw()

    def SaveDaAdded(self):
        peaks_x = self.da_added[:-1]
        peaks_x = peaks_x[:, 0].astype(np.int)
        self.peaks = self.da_added[:-1]
        self.peaks = self.peaks[:, 1]
        minima_x = self.da_minima_points_added[:-1]
        minima_x = minima_x[:, 0].astype(np.int)
        self.minima_peaks = self.da_minima_points_added[:-1]
        self.minima_peaks = self.minima_peaks[:, 1]
        self.added_da_prominence = peak_prominences(self.data, peaks_x)[0]
        self.added_da_width = peak_widths(self.data, peaks_x, rel_height=0.5)
        self.added_da_full_width = np.array([minima_x - self.added_da_width[2]]).flatten()
        return self.peaks, self.added_da_prominence, self.added_da_width[0], self.added_da_full_width, self.minima_peaks

    def SaveNonDaAdded(self):
        peaks_x = self.non_da_added[:-1]
        peaks_x = peaks_x[:, 0].astype(np.int)
        self.peaks = self.non_da_added[:-1]
        self.peaks = self.peaks[:, 1]
        minima_x  = self.non_da_minima_points_added[:-1]
        minima_x = minima_x[:,0].astype(np.int)
        self.minima_peaks = self.non_da_minima_points_added[:-1]
        self.minima_peaks = self.minima_peaks[:,1]
        self.added_non_da_prominence = peak_prominences(self.data, peaks_x)[0]
        self.added_non_da_width = peak_widths(self.data, peaks_x, rel_height=0.5)
        self.added_non_da_full_width = np.array([minima_x - self.added_non_da_width[2]]).flatten()
        return self.peaks,self.added_non_da_prominence,self.added_non_da_width[0],self.added_non_da_full_width,self.minima_peaks

    def GetMeasurements(self):
        if self.non_da_plot != None:
            peaks_x = self.non_da_amplitude_dialog.non_da_peaks[:,0].astype(np.int)
            self.prominence_non_da, self.prominence_properties_non_da = peak_prominences(self.data,peaks_x)[0],peak_prominences(self.data,peaks_x)
            self.peaks_width_non_da = peak_widths(self.data, peaks_x,prominence_data=self.prominence_properties_non_da, rel_height=0.5)
            return self.prominence_non_da,self.peaks_width_non_da
        if self.da_plot != None:
            peaks_x = self.da_amplitude_dialog.da_peaks[:,0].astype(np.int)
            self.prominence_da, self.prominence_properties_da = peak_prominences(self.data,peaks_x)[0],peak_prominences(self.data,peaks_x)
            self.peaks_width_da = peak_widths(self.data, peaks_x,prominence_data=self.prominence_properties_da, rel_height=0.5)
            return self.prominence_da, self.peaks_width_da

    def SaveDataset(self):
        """Saves .ann dataset: automatically added peaks and those with changed class
        Labels: NON_DA = 0; DA = 1"""
        try:
            if self.non_da_plot != None:
                catalog = './Dataset'
                if not os.path.exists(catalog):
                    os.makedirs(catalog)
                peaks_y_values = self.non_da_amplitude_dialog.non_da_peaks[:, 1]
                non_da_labels = np.zeros(len(peaks_y_values)).astype(np.int)
                peaks_y_values_minima = self.non_da_amplitude_dialog.minima[:, 1]
                prominence, width = self.GetMeasurements()
                peaks_full_width = np.array([self.non_da_amplitude_dialog.minima[:, 0] - width[2]]).flatten()
                if self.da_plot_added == None:
                    data = [peaks_y_values, prominence,width[0],peaks_full_width, peaks_y_values_minima,non_da_labels]
                    with open('./Dataset/' + f'{self.file_name_no_extension[0]}' + '.ann', 'wb') as f:
                        for i in data:
                            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                        QMessageBox.information(self,'Information',f'Data saved succesfully to: {self.file_name_no_extension[0]}.ann')
                else:
                    da_peaks, da_amplitude, da_width, da_full_width, da_minima = self.SaveDaAdded()
                    da_labels = np.ones(len(da_peaks)).astype(np.int)
                    peaks_y_values = np.concatenate((peaks_y_values,da_peaks))
                    prominence = np.concatenate((prominence,da_amplitude))
                    width = np.concatenate((width[0],da_width))
                    peaks_full_width = np.concatenate((peaks_full_width,da_full_width))
                    peaks_y_values_minima = np.concatenate((peaks_y_values_minima,da_minima))
                    labels = np.concatenate((non_da_labels,da_labels))
                    data = [peaks_y_values, prominence,width,peaks_full_width, peaks_y_values_minima,labels]
                    with open('./Dataset/' + f'{self.file_name_no_extension[0]}' + '.ann', 'wb') as f:
                        for i in data:
                            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                        QMessageBox.information(self,'Information',f'Data saved succesfully to: {self.file_name_no_extension[0]}.ann')

            if self.da_plot != None:
                catalog = './Dataset'
                if not os.path.exists(catalog):
                    os.makedirs(catalog)
                peaks_y_values = self.da_amplitude_dialog.da_peaks[:, 1]
                da_labels = np.ones(len(peaks_y_values)).astype(np.int)
                peaks_y_values_minima = self.da_amplitude_dialog.minima[:, 1]
                prominence, width = self.GetMeasurements()
                peaks_full_width = np.array([self.da_amplitude_dialog.minima[:, 0] - width[2]]).flatten()
                if self.non_da_plot_added == None:
                    data = [peaks_y_values, prominence,width[0],peaks_full_width, peaks_y_values_minima,da_labels]
                    with open('./Dataset/' + f'{self.file_name_no_extension[0]}' + '.ann', 'wb') as f:
                        for i in data:
                            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                        QMessageBox.information(self,'Information',f'Data saved succesfully to: {self.file_name_no_extension[0]}.ann')
                else:
                    non_da_peaks, non_da_amplitude, non_da_width, non_da_full_width, non_da_minima = self.SaveNonDaAdded()
                    non_da_labels = np.zeros(len(non_da_peaks)).astype(np.int)
                    peaks_y_values = np.concatenate((peaks_y_values, non_da_peaks))
                    prominence = np.concatenate((prominence, non_da_amplitude))
                    width = np.concatenate((width[0], non_da_width))
                    peaks_full_width = np.concatenate((peaks_full_width, non_da_full_width))
                    peaks_y_values_minima = np.concatenate((peaks_y_values_minima, non_da_minima))
                    labels = np.concatenate((da_labels, non_da_labels))
                    data = [peaks_y_values,prominence,width,peaks_full_width,peaks_y_values_minima,labels]
                    with open('./Dataset/' + f'{self.file_name_no_extension[0]}' + '.ann', 'wb') as f:
                        for i in data:
                            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                        QMessageBox.information(self, 'Information',f'Data saved succesfully to: {self.file_name_no_extension[0]}.ann')

        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            QMessageBox.critical(self,'Error',f'Something went wrong: {e}')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = TrainingPlot()
    w.show()
    sys.exit(app.exec_())