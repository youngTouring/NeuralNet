from PyQt5.QtWidgets import QDialog, QApplication, QTableWidgetItem, QTableWidget, QMessageBox
from Design.da_peaks_amplitude_tr import *
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
import sys

class DaPeaks(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonCalculate_da.clicked.connect(self.Calculate)
        self.da_peaks = np.empty(shape=2)
        self.peaks_full_width = None
        self.minima = None
        self.contour_heights = None
        self.da_peaks_properities = None
        self.minima_properties = None
        self.data = []

    def Calculate(self):
        try:
            peaks_height = self.ui.doubleSpinBox_da.value()
            self.da_peaks, self.da_peaks_properities = find_peaks(self.data, height=peaks_height)
            self.prominence, prominence = peak_prominences(self.data, self.da_peaks)[0], peak_prominences(self.data,self.da_peaks)
            self.peaks_width = peak_widths(self.data, self.da_peaks, prominence_data=prominence,rel_height=0.5)
            self.contour_heights = self.data[self.da_peaks] - self.prominence
            print(len(self.da_peaks))
            self.minima_data = self.data * (-1)
            self.minima, self.minima_properties = find_peaks(self.minima_data,distance=50, height=0.2)
            print(len(self.minima))
            self.prominence_minima, prominence_minima = peak_prominences(self.minima_data, self.minima)[0], \
                                                        peak_prominences(self.minima_data, self.minima)
            self.contour_heights_minima = self.minima_data[self.minima] - self.prominence_minima
            self.peaks_full_width = np.array([self.peaks_width[1],
                                              self.peaks_width[2],
                                              self.minima])
            self.ui.labelPeaksNumber_da.setText(f'Number of DA peaks: {len(self.da_peaks)}')

        except Exception as e:
            QMessageBox.information(self,'Info',e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DaPeaks()
    w.show()
    sys.exit(app.exec_())