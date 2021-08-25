from PyQt5.QtWidgets import QDialog, QApplication, QTableWidgetItem, QTableWidget, QMessageBox
from Design.non_da_peaks_amplitude_tr import *
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
import sys

class NonDaPeaks(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonCalculate_non_da.clicked.connect(self.Calculate)
        self.non_da_peaks = np.empty(shape=2)
        self.minima = None
        self.contour_heights = None
        self.peaks_full_width = None
        self.non_da_properties = None
        self.minima_properties = None
        self.data = []

    def Calculate(self):
        try:
            peaks_height = self.ui.doubleSpinBox_non_da.value()
            self.non_da_peaks, self.non_da_properties = find_peaks(self.data, height=peaks_height)
            self.prominence, prominence = peak_prominences(self.data, self.non_da_peaks)[0],peak_prominences(self.data,self.non_da_peaks)
            self.peaks_width = peak_widths(self.data, self.non_da_peaks, prominence_data=prominence,rel_height=0.5)
            self.contour_heights = self.data[self.non_da_peaks] - self.prominence

            self.minima_data = self.data * (-1)
            self.minima, self.minima_properties = find_peaks(self.minima_data, height=0.2,distance=200)
            print(len(self.minima))
            print(len(self.non_da_peaks))
            self.prominence_minima, prominence_minima = peak_prominences(self.minima_data, self.minima)[0], \
                                                        peak_prominences(self.minima_data, self.minima)
            self.contour_heights_minima = self.minima_data[self.minima] - self.prominence_minima
            self.peaks_full_width = np.array([self.peaks_width[1],
                                              self.peaks_width[2],
                                              self.minima])
            self.ui.labelPeaksNumber_non_da.setText(f'Number of NON_DA peaks: {len(self.non_da_peaks)}')

        except Exception as e:
            QMessageBox.critical(self,'Info',e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DaPeaks()
    w.show()
    sys.exit(app.exec_())