from PyQt5.QtWidgets import QMainWindow, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PlotApp.Design.train_plot_translated import *
from da_peaks_amplitude import *
from non_da_peaks_amplitude import *
import tensorflow as tf
from scipy.signal import savgol_filter, peak_widths, peak_prominences
import traceback
import h5py
import sys

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

        self.non_da = np.empty(shape=2)
        self.non_da_peak_heights_tab = []
        self.non_da_minima_heights_tab = []
        self.non_da_peak_heights = {}
        self.non_da_minima_heights = {}

        self.da_minima_heights_tab = []
        self.da_peak_heights_tab = []
        self.da_minima_heights = {}
        self.da = np.empty(shape=2)
        self.da_peak_heights = {}
        self.new_peaks_width = None
        self.new_peaks_width_left = None

        self.minima_da_plot = None
        self.minima_da_plot_added = None
        self.minima_da = np.empty(shape=2)
        self.minima_non_da = np.empty(shape=2)
        self.minima_non_da_plot = None
        self.minima_non_da_plot_added = None
        self.da_plot = None
        self.da_plot_added = None
        self.non_da_plot = None
        self.non_da_plot_added = None
        self.data_plot_combined = None

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
        self.non_da = np.empty(shape=2)
        self.non_da_peak_heights_tab = []
        self.non_da_minima_heights_tab = []
        self.non_da_peak_heights = {}
        self.non_da_minima_heights = {}

        self.da_minima_heights_tab = []
        self.da_peak_heights_tab = []
        self.da_minima_heights = {}
        self.da = np.empty(shape=2)
        self.da_peak_heights = {}
        self.new_peaks_width = None
        self.new_peaks_width_left = None

        self.minima_da_plot = None
        self.minima_da_plot_added = None
        self.minima_da = np.empty(shape=2)
        self.minima_non_da = np.empty(shape=2)
        self.minima_non_da_plot = None
        self.minima_non_da_plot_added = None
        self.da_plot = None
        self.da_plot_added = None
        self.non_da_plot = None
        self.non_da_plot_added = None
        self.data_plot_combined = None

        file, _ = QFileDialog.getOpenFileName(self,"Open files","",
                                              "All files (*);;H5 files (*.h5)")
        if file:
            try:
                ### Plot and data
                data = h5py.File(file, 'r')
                self.data = data.get('dataset_1')
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
                self.data_plot_combined = np.stack((self.xdataPlot, self.ydataPlot), axis=1)
                file_name = str(file).split('/')
                file_name_no_extension = file_name[-1].split('.')
                self.sc.axes.set_title(file_name_no_extension[0])
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
            self.da_plot = self.sc.axes.scatter(self.da_amplitude_dialog.da_peaks, self.data[self.da_amplitude_dialog.da_peaks],
                                                   color='red')
            self.minima_da_plot = self.sc.axes.scatter(self.da_amplitude_dialog.minima, self.data[self.da_amplitude_dialog.minima],
                                                       color='green')
            self.sc.axes.hlines(*self.da_amplitude_dialog.peaks_width[1:], color='C2', alpha=0.9)
            self.sc.axes.vlines(x=self.da_amplitude_dialog.da_peaks, ymin=self.da_amplitude_dialog.contour_heights,
                                ymax=self.data[self.da_amplitude_dialog.da_peaks], color='C3', alpha=0.4)
            self.sc.axes.vlines(x=self.da_amplitude_dialog.minima, ymin=self.da_amplitude_dialog.contour_heights_minima,
                                                 ymax=0, color='C3', alpha=0.4)
            self.sc.axes.hlines(*self.da_amplitude_dialog.peaks_full_width,color='C4', alpha=0.6, linestyles='--')
            self.da_amplitude_dialog.da_peaks = np.vstack((self.da_amplitude_dialog.da_peaks,
                                                           self.data[self.da_amplitude_dialog.da_peaks])).T
            self.da_amplitude_dialog.minima = np.vstack((self.da_amplitude_dialog.minima,self.data[self.da_amplitude_dialog.minima])).T
            self.new_peaks_width_left = self.da_amplitude_dialog.peaks_width[2]
            self.new_peaks_width = self.da_amplitude_dialog.peaks_width[0]
            self.da_amplitude_dialog.close()
        except Exception as e:
            QMessageBox.critical(self,'Error',e)

    def AddNonDaPeaks(self):
        try:
            self.non_da_plot = self.sc.axes.scatter(self.non_da_amplitude_dialog.non_da_peaks,
                                                   self.data[self.non_da_amplitude_dialog.non_da_peaks],
                                                   color='black')
            self.minima_non_da_plot = self.sc.axes.scatter(self.non_da_amplitude_dialog.minima, self.data[self.non_da_amplitude_dialog.minima],
                                                           color='blue')
            self.sc.axes.hlines(*self.non_da_amplitude_dialog.peaks_width[1:], color='C2', alpha=0.4)
            self.sc.axes.vlines(x=self.non_da_amplitude_dialog.non_da_peaks, ymin=self.non_da_amplitude_dialog.contour_heights,
                                ymax=self.data[self.non_da_amplitude_dialog.non_da_peaks], color='C3', alpha=0.4)
            self.sc.axes.vlines(x=self.non_da_amplitude_dialog.minima,
                                                 ymin=self.non_da_amplitude_dialog.contour_heights_minima,
                                                 ymax=0, color='C3', alpha=0.4)
            self.sc.axes.hlines(*self.non_da_amplitude_dialog.peaks_full_width,color='C4',alpha = 0.6, linestyles='--')
            self.non_da_amplitude_dialog.non_da_peaks = np.vstack((self.non_da_amplitude_dialog.non_da_peaks,
                                                                   self.data[self.non_da_amplitude_dialog.non_da_peaks])).T
            self.non_da_amplitude_dialog.minima = np.vstack((self.non_da_amplitude_dialog.minima,
                                                             self.data[self.non_da_amplitude_dialog.minima])).T
            self.new_peaks_width_left = self.non_da_amplitude_dialog.peaks_width[2]
            self.new_peaks_width = self.non_da_amplitude_dialog.peaks_width[0]
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
                        new_prominence = self.non_da_amplitude_dialog.prominence
                        new_properties = self.non_da_amplitude_dialog.non_da_properties.get('peak_heights')
                        new_properties_minima = self.non_da_amplitude_dialog.minima_properties.get('peak_heights')
                        value_to_delete = np.where(self.non_da_amplitude_dialog.non_da_peaks == i)
                        value_to_add = value_to_delete[0].item(0)
                        self.da_peak_heights_tab.append(new_properties[value_to_add])
                        point_to_add = self.non_da_amplitude_dialog.non_da_peaks[value_to_add]
                        self.da = np.insert(self.da, 0, point_to_add, axis=0)
                        self.da = np.reshape(self.da, (-1, 2))
                        self.da_peak_heights = {'peak_heights': np.array(self.da_peak_heights_tab)}
                        new_prominence = np.delete(new_prominence,np.where(new_prominence == new_prominence[value_to_delete[0]]),axis=0)
                        self.new_peaks_width = np.delete(self.new_peaks_width,
                                                           np.where(self.new_peaks_width == self.new_peaks_width[value_to_delete[0]]),
                                                           axis=0)
                        self.new_peaks_width_left = np.delete(self.new_peaks_width_left,
                                                         np.where(self.new_peaks_width_left == self.new_peaks_width_left[value_to_delete[0]]),axis=0)
                        new_properties = np.delete(new_properties,
                                                   np.where(new_properties == new_properties[value_to_delete[0]]),axis=0)
                        self.non_da_amplitude_dialog.prominence = new_prominence
                        self.non_da_amplitude_dialog.non_da_properties['peak_heights'] = new_properties
                        self.non_da_amplitude_dialog.non_da_peaks = np.delete(
                            self.non_da_amplitude_dialog.non_da_peaks, np.where(self.non_da_amplitude_dialog.non_da_peaks[:, 0] == i),
                            axis=0)
                        self.non_da_amplitude_dialog.non_da_peaks = np.reshape(self.non_da_amplitude_dialog.non_da_peaks, (-1, 2))
                        self.non_da_plot.set_offsets(self.non_da_amplitude_dialog.non_da_peaks)
                        self.da_minima_heights_tab.append(new_properties_minima[value_to_add])
                        minima_point_to_add = self.non_da_amplitude_dialog.minima[value_to_add]
                        self.minima_da = np.insert(self.minima_da, 0, minima_point_to_add, axis=0)
                        self.minima_da = np.reshape(self.minima_da, (-1, 2))
                        self.da_minima_heights = {'peak_heights': np.array(self.da_peak_heights_tab)}
                        value_to_delete_minima = value_to_delete[0].item(0)
                        new_properties_minima = np.delete(new_properties_minima,
                                                          np.where(new_properties_minima[value_to_delete[0]]),axis=0)
                        self.non_da_amplitude_dialog.minima_properties['peak_heights'] = new_properties_minima
                        self.non_da_amplitude_dialog.minima = np.delete(self.non_da_amplitude_dialog.minima,value_to_delete_minima,axis=0)
                        self.non_da_amplitude_dialog.minima = np.reshape(self.non_da_amplitude_dialog.minima, (-1, 2))
                        self.minima_non_da_plot.set_offsets(self.non_da_amplitude_dialog.minima)
                        self.Update_da_plot()
                        break
                if self.da_plot != None:
                    if i in self.da_amplitude_dialog.da_peaks[:,0]:
                        new_prominence = self.da_amplitude_dialog.prominence
                        new_properties = self.da_amplitude_dialog.da_peaks_properities.get('peak_heights')
                        new_properties_minima = self.da_amplitude_dialog.minima_properties.get('peak_heights')
                        value_to_delete = np.where(self.da_amplitude_dialog.da_peaks == i)
                        value_to_add = value_to_delete[0].item(0)
                        self.non_da_peak_heights_tab.append(new_properties[value_to_add])
                        point_to_add = self.da_amplitude_dialog.da_peaks[value_to_add]
                        self.non_da = np.insert(self.non_da, 0,point_to_add, axis=0)
                        self.non_da = np.reshape(self.non_da, (-1, 2))
                        self.non_da_peak_heights = {'peak_heights': np.array(self.non_da_peak_heights_tab)}
                        new_prominence = np.delete(new_prominence,
                                                   np.where(new_prominence == new_prominence[value_to_delete[0]]),
                                                   axis=0)
                        self.new_peaks_width = np.delete(self.new_peaks_width,
                                                    np.where(self.new_peaks_width == self.new_peaks_width[value_to_delete[0]]),
                                                    axis=0)
                        self.new_peaks_width_left = np.delete(self.new_peaks_width_left,
                                                         np.where(self.new_peaks_width_left == self.new_peaks_width_left[
                                                             value_to_delete[0]]), axis=0)
                        new_properties = np.delete(new_properties,np.where(new_properties == new_properties[value_to_delete[0]]),axis=0)
                        self.da_amplitude_dialog.prominence = new_prominence
                        self.da_amplitude_dialog.da_peaks_properities['peak_heights'] = new_properties
                        self.da_amplitude_dialog.da_peaks = np.delete(
                            self.da_amplitude_dialog.da_peaks,np.where(self.da_amplitude_dialog.da_peaks[:, 0] == i),axis=0)
                        self.da_amplitude_dialog.da_peaks = np.reshape(self.da_amplitude_dialog.da_peaks,(-1, 2))
                        self.da_plot.set_offsets(self.da_amplitude_dialog.da_peaks)
                        self.non_da_minima_heights_tab.append(new_properties_minima[value_to_add])
                        minima_point_to_add = self.da_amplitude_dialog.minima[value_to_add]
                        self.minima_non_da = np.insert(self.minima_non_da,0,minima_point_to_add,axis=0)
                        self.minima_non_da = np.reshape(self.minima_non_da, (-1, 2))
                        self.non_da_minima_heights = {'peak_heights': np.array(self.non_da_peak_heights_tab)}
                        value_to_delete_minima = value_to_delete[0].item(0)
                        new_properties_minima = np.delete(new_properties_minima,
                                                          np.where(new_properties_minima[value_to_delete[0]]),axis=0)
                        self.da_amplitude_dialog.minima_properties['peak_heights'] = new_properties_minima
                        self.da_amplitude_dialog.minima = np.delete(self.da_amplitude_dialog.minima,value_to_delete_minima,axis=0)
                        self.da_amplitude_dialog.minima = np.reshape(self.da_amplitude_dialog.minima,(-1,2))
                        self.minima_da_plot.set_offsets(self.da_amplitude_dialog.minima)
                        self.Update_non_da_plot()
                        break

    def Update_non_da_plot(self):
        non_da = self.non_da[:-1]
        minima = self.minima_non_da[:-1]
        if self.non_da_plot_added == None:
            self.non_da_plot_added = self.sc.axes.scatter(non_da[:, 0], non_da[:, 1], color='black')
            self.minima_non_da_plot_added = self.sc.axes.scatter(minima[:,0],minima[:,1],color='blue')
        else:
            self.non_da_plot_added.set_offsets(non_da)
            self.minima_non_da_plot_added.set_offsets(minima)
            self.sc.draw()

    def Update_da_plot(self):
        da = self.da[:-1]
        minima = self.minima_da[:-1]
        if self.da_plot_added == None:
            self.da_plot_added = self.sc.axes.scatter(da[:, 0], da[:, 1], color='red')
            self.minima_da_plot_added = self.sc.axes.scatter(minima[:, 0], minima[:, 1], color='green')
        else:
            self.da_plot_added.set_offsets(da)
            self.minima_da_plot_added.set_offsets(minima)
            self.sc.draw()

    def SaveDaAdded(self):
        peaks_x = self.da[:-1]
        peaks_x = peaks_x[:, 0].astype(np.int)
        self.peaks = self.da[:-1]
        self.peaks = self.peaks[:, 1]
        minima_x = self.minima_da[:-1]
        minima_x = minima_x[:, 0].astype(np.int)
        self.minima_peaks = self.minima_da[:-1]
        self.minima_peaks = self.minima_peaks[:, 1]
        self.added_da_prominence = peak_prominences(self.data, peaks_x)[0]
        self.added_da_width = peak_widths(self.data, peaks_x, rel_height=0.5)
        self.added_da_full_width = np.array([minima_x - self.added_da_width[2]]).flatten()
        return self.peaks, self.added_da_prominence, self.added_da_width[0], self.added_da_full_width, self.minima_peaks

    def SaveNonDaAdded(self):
        peaks_x = self.non_da[:-1]
        peaks_x = peaks_x[:, 0].astype(np.int)
        self.peaks = self.non_da[:-1]
        self.peaks = self.peaks[:, 1]
        minima_x  = self.minima_non_da[:-1]
        minima_x = minima_x[:,0].astype(np.int)
        self.minima_peaks = self.minima_non_da[:-1]
        self.minima_peaks = self.minima_peaks[:,1]
        self.added_non_da_prominence = peak_prominences(self.data, peaks_x)[0]
        self.added_non_da_width = peak_widths(self.data, peaks_x, rel_height=0.5)
        self.added_non_da_full_width = np.array([minima_x - self.added_non_da_width[2]]).flatten()
        return self.peaks,self.added_non_da_prominence,self.added_non_da_width[0],self.added_non_da_full_width,self.minima_peaks

    def SaveDataset(self):
        """Saves tf dataset automatically added peaks and those with changed class
        Labels: NON_DA = 0; DA = 1"""
        try:
            if self.non_da_plot != None:
                da_peaks, da_amplitude, da_width, da_full_width, da_minima = self.SaveDaAdded()
                da_labels = np.ones(len(da_peaks)).astype(np.int)
                peaks_y_values = self.non_da_amplitude_dialog.non_da_properties.get('peak_heights')
                non_da_labels = np.zeros(len(peaks_y_values)).astype(np.int)
                peaks_y_values_minima = self.non_da_amplitude_dialog.minima_properties.get('peak_heights')
                peaks_full_width = np.array([self.non_da_amplitude_dialog.minima[:, 0] - self.new_peaks_width_left]).flatten()
                DS_DA = tf.data.Dataset.from_tensor_slices((da_peaks, da_amplitude, da_width, da_full_width, da_minima,da_labels))
                DS_NON_DA = tf.data.Dataset.from_tensor_slices((peaks_y_values, self.non_da_amplitude_dialog.prominence
                                                               , self.new_peaks_width,
                                                               peaks_full_width, peaks_y_values_minima * -1,non_da_labels))
                DS_JOINT = DS_NON_DA.concatenate(DS_DA)
                # path = 'C:/Users/miko5/Desktop/Python/ML/Deep_learning/Scikitlearn/PlotApp'
                # tf.data.experimental.save(DS_JOINT, path=path, compression=None, shard_func=None)
            if self.da_plot != None:
                non_da_peaks,non_da_amplitude,non_da_width,non_da_full_width,non_da_minima = self.SaveNonDaAdded()
                non_da_labels = np.zeros(len(non_da_peaks)).astype(np.int)
                DS_NON_DA = tf.data.Dataset.from_tensor_slices((non_da_peaks,non_da_amplitude,non_da_width,
                                                                      non_da_full_width,non_da_minima,non_da_labels))
                peaks_y_values = self.da_amplitude_dialog.da_peaks_properities.get('peak_heights')
                da_labels = np.ones(len(peaks_y_values)).astype(np.int)
                peaks_y_values_minima = self.da_amplitude_dialog.minima_properties.get('peak_heights')
                peaks_full_width = np.array([self.da_amplitude_dialog.minima[:, 0] - self.new_peaks_width_left]).flatten()
                DS_DA = tf.data.Dataset.from_tensor_slices((peaks_y_values,self.da_amplitude_dialog.prominence
                                                               ,self.new_peaks_width,
                                                               peaks_full_width,peaks_y_values_minima*-1,da_labels))
                DS_JOINT = DS_DA.concatenate(DS_NON_DA)
                # path = 'C:/Users/miko5/Desktop/Python/ML/Deep_learning/Scikitlearn/PlotApp'
                # tf.data.experimental.save(DS_JOINT, path=path, compression=None, shard_func=None)
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            QMessageBox.critical(self,'Error',f'Something went wrong: {e}')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = TrainingPlot()
    w.show()
    sys.exit(app.exec_())