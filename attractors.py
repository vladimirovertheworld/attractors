import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QTextEdit, QPushButton
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

class AttractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Strange Attractors")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.plot_widget = gl.GLViewWidget()
        self.layout.addWidget(self.plot_widget, 2)

        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout, 1)

        self.attractor_combo = QComboBox()
        self.attractors = {
            "Lorenz": self.lorenz,
            "Rössler": self.rossler,
            "Aizawa": self.aizawa,
            "Chen": self.chen,
            "Halvorsen": self.halvorsen,
            "Thomas": self.thomas,
            "Sprott": self.sprott,
            "Dadras": self.dadras,
            "Four-Wing": self.four_wing,
            "Burke-Shaw": self.burke_shaw
        }
        self.attractor_combo.addItems(self.attractors.keys())
        self.control_layout.addWidget(self.attractor_combo)

        self.description = QTextEdit()
        self.description.setReadOnly(True)
        self.control_layout.addWidget(self.description)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_animation)
        self.control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_animation)
        self.control_layout.addWidget(self.stop_button)

        self.attractor_combo.currentTextChanged.connect(self.update_description)
        self.update_description(self.attractor_combo.currentText())

        self.points = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=2)
        self.plot_widget.addItem(self.points)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.current_attractor = None
        self.x, self.y, self.z = 0.1, 0.1, 0.1

    def start_animation(self):
        self.current_attractor = self.attractors[self.attractor_combo.currentText()]
        self.x, self.y, self.z = 0.1, 0.1, 0.1
        self.timer.start(50)

    def stop_animation(self):
        self.timer.stop()

    def update_plot(self):
        points = np.zeros((1000, 3))
        for i in range(1000):
            self.x, self.y, self.z = self.current_attractor(self.x, self.y, self.z)
            points[i] = [self.x, self.y, self.z]
        self.points.setData(pos=points, color=(1, 1, 1, 1), size=2)

    def update_description(self, attractor_name):
        descriptions = {
            "Lorenz": "The Lorenz attractor is a classic example of a chaotic system. It was first studied by Edward Lorenz in 1963 and is known for its butterfly-like shape.\nMore info: https://arxiv.org/abs/nlin/0501023",
            "Rössler": "The Rössler attractor was discovered by Otto Rössler in 1976. It's known for its simple equations but complex behavior.\nMore info: https://arxiv.org/abs/nlin/0502028",
            "Aizawa": "The Aizawa attractor is a lesser-known but visually striking attractor with a unique spiral structure.\nMore info: https://arxiv.org/abs/1101.2124",
            "Chen": "The Chen system is a three-dimensional flow that exhibits chaotic behavior. It was discovered by Guanrong Chen and Tetsushi Ueta in 1999.\nMore info: https://arxiv.org/abs/nlin/0307009",
            "Halvorsen": "The Halvorsen attractor is known for its symmetrical, pretzel-like shape and was discovered by Norwegian mathematician Ingemar Halvorsen.\nMore info: https://arxiv.org/abs/1007.1057",
            "Thomas": "The Thomas' cyclically symmetric attractor was discovered by René Thomas. It's known for its highly symmetric structure.\nMore info: https://arxiv.org/abs/nlin/0107044",
            "Sprott": "The Sprott attractor is one of many chaotic systems discovered by Julien C. Sprott through a systematic search of simple chaotic flows.\nMore info: https://arxiv.org/abs/nlin/0507037",
            "Dadras": "The Dadras system is a relatively new chaotic system with a unique butterfly-like structure.\nMore info: https://arxiv.org/abs/1008.4044",
            "Four-Wing": "The Four-Wing attractor is a chaotic system that exhibits a unique four-wing butterfly shape.\nMore info: https://arxiv.org/abs/1301.0538",
            "Burke-Shaw": "The Burke-Shaw attractor is a three-dimensional chaotic attractor discovered by Bill Burke and Robert Shaw in 1981.\nMore info: https://arxiv.org/abs/nlin/0301023"
        }
        self.description.setText(descriptions[attractor_name])

    # Attractor equations
    def lorenz(self, x, y, z, a=10, b=28, c=8/3):
        dx = a * (y - x)
        dy = x * (b - z) - y
        dz = x * y - c * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def rossler(self, x, y, z, a=0.2, b=0.2, c=5.7):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def aizawa(self, x, y, z, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def chen(self, x, y, z, a=35, b=3, c=28):
        dx = a * (y - x)
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def halvorsen(self, x, y, z, a=1.4):
        dx = -a * x - 4 * y - 4 * z - y**2
        dy = -a * y - 4 * z - 4 * x - z**2
        dz = -a * z - 4 * x - 4 * y - x**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def thomas(self, x, y, z, b=0.208186):
        dx = -b * x + np.sin(y)
        dy = -b * y + np.sin(z)
        dz = -b * z + np.sin(x)
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def sprott(self, x, y, z, a=2.07):
        dx = y + a * x * y + x * z
        dy = 1 - a * x**2 + y * z
        dz = x - x**2 - y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def dadras(self, x, y, z, a=3, b=2.7, c=1.7, d=2, e=9):
        dx = y - a * x + b * y * z
        dy = c * y - x * z + z
        dz = d * x * y - e * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def four_wing(self, x, y, z, a=0.2, b=0.01, c=-0.4):
        dx = a * x + y * z
        dy = b * x + c * y - x * z
        dz = -z - x * y
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def burke_shaw(self, x, y, z, s=10, v=4.272):
        dx = -s * (x + y)
        dy = -y - s * x * z
        dz = s * x * y + v
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AttractorApp()
    window.show()
    sys.exit(app.exec())