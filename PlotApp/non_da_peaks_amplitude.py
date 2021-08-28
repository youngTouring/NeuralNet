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
        self.non_da_properties = None
        self.data = []

    def Calculate(self):
        try:
            peaks_height = self.ui.doubleSpinBox_non_da.value()
            self.non_da_peaks, self.non_da_properties = find_peaks(self.data, height=peaks_height)
            minima_points_y = []
            minima_points_x = []
            for i,j in zip(self.non_da_peaks,self.non_da_peaks + 100):
                point_y = np.min(self.data[i:j])
                point_x = np.where(self.data == point_y)
                minima_points_x.append(point_x[0].item(0))
                minima_points_y.append(point_y)
            minima_y = np.array(minima_points_y)
            minima_x = np.array(minima_points_x)
            self.non_da_peaks = np.vstack((self.non_da_peaks,self.data[self.non_da_peaks])).T
            self.minima = np.vstack((minima_x,minima_y)).T
            self.ui.labelPeaksNumber_non_da.setText(f'Number of NON_DA peaks: {len(self.non_da_peaks)}')
        except Exception as e:
            QMessageBox.critical(self,'Info',e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = NonDaPeaks()
    w.show()
    sys.exit(app.exec_())