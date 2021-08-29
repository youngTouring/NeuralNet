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
        self.minima = None
        self.da_peaks_properities = None
        self.data = []

    def Calculate(self):
        try:
            peaks_height = self.ui.doubleSpinBox_da.value()
            self.da_peaks, self.da_peaks_properities = find_peaks(self.data, height=peaks_height,distance=150)
            minima_points_x = []
            minima_points_y = []
            for i,j in zip(self.da_peaks,self.da_peaks + 100):
                point_y = np.min(self.data[i:j])
                point_x = np.where(self.data == point_y)
                minima_points_x.append(point_x[0].item(0))
                minima_points_y.append(point_y)
            minima_x = np.array(minima_points_x)
            minima_y = np.array(minima_points_y)
            self.da_peaks = np.vstack((self.da_peaks, self.data[self.da_peaks])).T
            self.minima = np.vstack((minima_x, minima_y)).T
            self.ui.labelPeaksNumber_da.setText(f'Number of DA peaks: {len(self.da_peaks)}')
        except Exception as e:
            QMessageBox.information(self,'Info',e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DaPeaks()
    w.show()
    sys.exit(app.exec_())