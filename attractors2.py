import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QTextEdit, QPushButton, QSlider
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg

class AttractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Strange Attractors")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.plot_widget = gl.GLViewWidget()
        self.layout.addWidget(self.plot_widget, 2)

        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout, 1)

        self.attractor_combo = QComboBox()
        self.attractors = {
            "Lorenz": (self.lorenz, ["sigma", "rho", "beta"]),
            "Rössler": (self.rossler, ["a", "b", "c"]),
            "Aizawa": (self.aizawa, ["a", "b", "c", "d", "e", "f"]),
            "Chen": (self.chen, ["a", "b", "c"]),
            "Halvorsen": (self.halvorsen, ["a"]),
            "Thomas": (self.thomas, ["b"]),
            "Sprott": (self.sprott, ["a"]),
            "Dadras": (self.dadras, ["a", "b", "c", "d", "e"]),
            "Four-Wing": (self.four_wing, ["a", "b", "c"]),
            "Burke-Shaw": (self.burke_shaw, ["s", "v"]),
            "Lorenz83": (self.lorenz83, ["a", "b", "f", "g"]),
            "Moore-Spiegel": (self.moore_spiegel, ["a", "b", "c"]),
            "Rucklidge": (self.rucklidge, ["a", "k"]),
            "Dequan Li": (self.dequan_li, ["a", "c", "d", "e", "k", "f"]),
            "Yu-Wang": (self.yu_wang, ["a", "b", "c", "d"]),
            "Nose-Hoover": (self.nose_hoover, ["a"]),
            "Rabinovich-Fabrikant": (self.rabinovich_fabrikant, ["alpha", "gamma"]),
            "Three-Scroll Unified Chaotic System": (self.three_scroll, ["a", "b", "c", "d", "e"]),
            "Tamari": (self.tamari, ["a", "b", "c"]),
            "Scroll": (self.scroll, ["a", "b", "c", "d"])
        }
        self.attractor_combo.addItems(self.attractors.keys())
        self.control_layout.addWidget(self.attractor_combo)

        self.description = QTextEdit()
        self.description.setReadOnly(True)
        self.control_layout.addWidget(self.description)

        self.sliders = {}
        self.slider_layout = QVBoxLayout()
        self.control_layout.addLayout(self.slider_layout)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_animation)
        self.control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_animation)
        self.control_layout.addWidget(self.stop_button)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        self.control_layout.addWidget(self.quit_button)

        self.attractor_combo.currentTextChanged.connect(self.update_description)
        self.update_description(self.attractor_combo.currentText())

        self.points = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=pg.glColor((255, 0, 0)), width=1.5, antialias=True)
        self.plot_widget.addItem(self.points)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.current_attractor = None
        self.x, self.y, self.z = 0.1, 0.1, 0.1

    def create_sliders(self, params):
        for widget in self.sliders.values():
            self.slider_layout.removeWidget(widget)
            widget.deleteLater()
        self.sliders.clear()

        for param in params:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            self.sliders[param] = slider
            self.slider_layout.addWidget(slider)

    def start_animation(self):
        attractor_name = self.attractor_combo.currentText()
        self.current_attractor, params = self.attractors[attractor_name]
        self.create_sliders(params)
        self.x, self.y, self.z = 0.1, 0.1, 0.1
        self.points_list = np.array([[self.x, self.y, self.z]])
        self.timer.start(50)

    def stop_animation(self):
        self.timer.stop()

    def update_plot(self):
        attractor_name = self.attractor_combo.currentText()
        for _ in range(10):
            params = [slider.value() / 50 for slider in self.sliders.values()]
            self.x, self.y, self.z = self.current_attractor(self.x, self.y, self.z, *params)
            self.points_list = np.vstack((self.points_list, [self.x, self.y, self.z]))

        colors = np.array([pg.glColor((i, self.points_list.shape[0])) for i in range(self.points_list.shape[0])])
        self.points.setData(pos=self.points_list, color=colors)

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
            "Burke-Shaw": "The Burke-Shaw attractor is a three-dimensional chaotic attractor discovered by Bill Burke and Robert Shaw in 1981.\nMore info: https://arxiv.org/abs/nlin/0301023",
            "Lorenz83": "The Lorenz-83 model is a simplified model of the general atmospheric circulation.\nMore info: https://arxiv.org/abs/nlin/0110046",
            "Moore-Spiegel": "The Moore-Spiegel oscillator is a model for the dynamo of a star.\nMore info: https://arxiv.org/abs/1409.3554",
            "Rucklidge": "The Rucklidge attractor models thermal convection in fluids.\nMore info: https://arxiv.org/abs/nlin/0508021",
            "Dequan Li": "The Dequan Li system is a new chaotic attractor with a complex structure.\nMore info: https://arxiv.org/abs/1205.3181",
            "Yu-Wang": "The Yu-Wang system is a recently discovered chaotic system with a unique structure.\nMore info: https://arxiv.org/abs/1406.5353",
            "Nose-Hoover": "The Nose-Hoover oscillator is used in molecular dynamics simulations.\nMore info: https://arxiv.org/abs/1609.00767",
            "Rabinovich-Fabrikant": "The Rabinovich-Fabrikant equations model the stochasticity in a plasma.\nMore info: https://arxiv.org/abs/1604.02081",
            "Three-Scroll Unified Chaotic System": "This system unifies several known chaotic systems and produces three-scroll attractors.\nMore info: https://arxiv.org/abs/1404.2548",
            "Tamari": "The Tamari attractor is a lesser-known chaotic system with interesting properties.\nMore info: https://arxiv.org/abs/1706.05364",
            "Scroll": "The Scroll system produces a scroll-shaped attractor.\nMore info: https://arxiv.org/abs/1510.00344"
        }
        self.description.setText(descriptions[attractor_name])

    # Attractor equations
    def lorenz(self, x, y, z, sigma=10, rho=28, beta=8/3):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
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

    def lorenz83(self, x, y, z, a=0.95, b=7.91, f=4.83, g=4.66):
        dx = -a * x - y**2 - z**2 + a * f
        dy = -y + x * y - b * x * z + g
        dz = -z + b * x * y + x * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def moore_spiegel(self, x, y, z, a=100, b=26, c=0.5):
        dx = y
        dy = z
        dz = -z - (a - c * x**2) * y - c * x
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01
    
    def rucklidge(self, x, y, z, a=2, k=6.7):
        dx = -a * x + k * y - y * z
        dy = x
        dz = -z + y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def dequan_li(self, x, y, z, a=40, c=1.833, d=0.16, e=0.65, k=55, f=20):
        dx = a * (y - x) + d * x * z
        dy = k * x + f * y - x * z
        dz = c * z + x * y - e * x**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def yu_wang(self, x, y, z, a=10, b=40, c=2, d=2.5):
        dx = a * (y - x)
        dy = b * x - c * x * z
        dz = np.exp(x * y) - d * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def nose_hoover(self, x, y, z, a=1.5):
        dx = y
        dy = -x + y * z
        dz = a - y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def rabinovich_fabrikant(self, x, y, z, alpha=0.14, gamma=0.10):
        dx = y * (z - 1 + x**2) + gamma * x
        dy = x * (3 * z + 1 - x**2) + gamma * y
        dz = -2 * z * (alpha + x * y)
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def three_scroll(self, x, y, z, a=40, b=0.833, c=20, d=0.5, e=0.65):
        dx = a * (y - x) + d * x * z
        dy = c * y - x * z
        dz = b * z + x * y - e * x**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def tamari(self, x, y, z, a=1.5, b=0.8, c=2.5):
        dx = y - a * x
        dy = b * x - y**2 - z**2
        dz = x * y - c * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def scroll(self, x, y, z, a=40, b=0.833, c=20, d=0.5):
        dx = a * (y - x) + d * x * z
        dy = c * y - x * z
        dz = b * z + x * y - y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01


    def create_sliders(self, params):
        for widget in self.sliders.values():
            self.slider_layout.removeWidget(widget)
            widget.deleteLater()
        self.sliders.clear()

        for param in params:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            self.sliders[param] = slider
            self.slider_layout.addWidget(slider)

    def start_animation(self):
        attractor_name = self.attractor_combo.currentText()
        self.current_attractor, params = self.attractors[attractor_name]
        self.create_sliders(params)
        self.x, self.y, self.z = 0.1, 0.1, 0.1
        self.points_list = np.array([[self.x, self.y, self.z]])
        self.timer.start(50)

    def stop_animation(self):
        self.timer.stop()

    def update_plot(self):
        attractor_name = self.attractor_combo.currentText()
        for _ in range(10):
            params = [slider.value() / 50 for slider in self.sliders.values()]
            self.x, self.y, self.z = self.current_attractor(self.x, self.y, self.z, *params)
            self.points_list = np.vstack((self.points_list, [self.x, self.y, self.z]))

        colors = np.array([pg.glColor((i, self.points_list.shape[0])) for i in range(self.points_list.shape[0])])
        self.points.setData(pos=self.points_list, color=colors)

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
            "Burke-Shaw": "The Burke-Shaw attractor is a three-dimensional chaotic attractor discovered by Bill Burke and Robert Shaw in 1981.\nMore info: https://arxiv.org/abs/nlin/0301023",
            "Lorenz83": "The Lorenz-83 model is a simplified model of the general atmospheric circulation.\nMore info: https://arxiv.org/abs/nlin/0110046",
            "Moore-Spiegel": "The Moore-Spiegel oscillator is a model for the dynamo of a star.\nMore info: https://arxiv.org/abs/1409.3554",
            "Rucklidge": "The Rucklidge attractor models thermal convection in fluids.\nMore info: https://arxiv.org/abs/nlin/0508021",
            "Dequan Li": "The Dequan Li system is a new chaotic attractor with a complex structure.\nMore info: https://arxiv.org/abs/1205.3181",
            "Yu-Wang": "The Yu-Wang system is a recently discovered chaotic system with a unique structure.\nMore info: https://arxiv.org/abs/1406.5353",
            "Nose-Hoover": "The Nose-Hoover oscillator is used in molecular dynamics simulations.\nMore info: https://arxiv.org/abs/1609.00767",
            "Rabinovich-Fabrikant": "The Rabinovich-Fabrikant equations model the stochasticity in a plasma.\nMore info: https://arxiv.org/abs/1604.02081",
            "Three-Scroll Unified Chaotic System": "This system unifies several known chaotic systems and produces three-scroll attractors.\nMore info: https://arxiv.org/abs/1404.2548",
            "Tamari": "The Tamari attractor is a lesser-known chaotic system with interesting properties.\nMore info: https://arxiv.org/abs/1706.05364",
            "Scroll": "The Scroll system produces a scroll-shaped attractor.\nMore info: https://arxiv.org/abs/1510.00344"
        }
        self.description.setText(descriptions[attractor_name])

    # Attractor equations
    def lorenz(self, x, y, z, sigma=10, rho=28, beta=8/3):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
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

    def lorenz83(self, x, y, z, a=0.95, b=7.91, f=4.83, g=4.66):
        dx = -a * x - y**2 - z**2 + a * f
        dy = -y + x * y - b * x * z + g
        dz = -z + b * x * y + x * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def moore_spiegel(self, x, y, z, a=100, b=26, c=0.5):
        dx = y
        dy = z
        dz = -z - (a - c * x**2) * y - c * x
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01
    
    def rucklidge(self, x, y, z, a=2, k=6.7):
        dx = -a * x + k * y - y * z
        dy = x
        dz = -z + y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def dequan_li(self, x, y, z, a=40, c=1.833, d=0.16, e=0.65, k=55, f=20):
        dx = a * (y - x) + d * x * z
        dy = k * x + f * y - x * z
        dz = c * z + x * y - e * x**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def yu_wang(self, x, y, z, a=10, b=40, c=2, d=2.5):
        dx = a * (y - x)
        dy = b * x - c * x * z
        dz = np.exp(x * y) - d * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def nose_hoover(self, x, y, z, a=1.5):
        dx = y
        dy = -x + y * z
        dz = a - y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def rabinovich_fabrikant(self, x, y, z, alpha=0.14, gamma=0.10):
        dx = y * (z - 1 + x**2) + gamma * x
        dy = x * (3 * z + 1 - x**2) + gamma * y
        dz = -2 * z * (alpha + x * y)
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def three_scroll(self, x, y, z, a=40, b=0.833, c=20, d=0.5, e=0.65):
        dx = a * (y - x) + d * x * z
        dy = c * y - x * z
        dz = b * z + x * y - e * x**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def tamari(self, x, y, z, a=1.5, b=0.8, c=2.5):
        dx = y - a * x
        dy = b * x - y**2 - z**2
        dz = x * y - c * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01

    def scroll(self, x, y, z, a=40, b=0.833, c=20, d=0.5):
        dx = a * (y - x) + d * x * z
        dy = c * y - x * z
        dz = b * z + x * y - y**2
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AttractorApp()
    window.show()
    sys.exit(app.exec())