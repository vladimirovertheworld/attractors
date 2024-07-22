import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QListWidget, QMessageBox
from PyQt6.QtCore import Qt

# Attractor functions (20 examples)
def lorenz(t, sigma=10, rho=28, beta=8/3):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 1, 1, 1
    for i in range(1, t.size):
        x[i] = x[i-1] + sigma * (y[i-1] - x[i-1]) * dt
        y[i] = y[i-1] + (x[i-1] * (rho - z[i-1]) - y[i-1]) * dt
        z[i] = z[i-1] + (x[i-1] * y[i-1] - beta * z[i-1]) * dt
    return x, y, z

def rossler(t, a=0.2, b=0.2, c=5.7):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 1, 1, 1
    for i in range(1, t.size):
        x[i] = x[i-1] + (-y[i-1] - z[i-1]) * dt
        y[i] = y[i-1] + (x[i-1] + a * y[i-1]) * dt
        z[i] = z[i-1] + (b + z[i-1] * (x[i-1] - c)) * dt
    return x, y, z

def thomas(t, b=0.208186):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 1, 1, 1
    for i in range(1, t.size):
        x[i] = x[i-1] + (np.sin(y[i-1]) - b * x[i-1]) * dt
        y[i] = y[i-1] + (np.sin(z[i-1]) - b * y[i-1]) * dt
        z[i] = z[i-1] + (np.sin(x[i-1]) - b * z[i-1]) * dt
    return x, y, z

def aizawa(t, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0, 0
    for i in range(1, t.size):
        x[i] = x[i-1] + ((z[i-1] - b) * x[i-1] - d * y[i-1]) * dt
        y[i] = y[i-1] + (d * x[i-1] + (z[i-1] - b) * y[i-1]) * dt
        z[i] = z[i-1] + (c + a * z[i-1] - (z[i-1]**3) / 3 - (x[i-1]**2 + y[i-1]**2) * (1 + e * z[i-1]) + f * z[i-1] * (x[i-1]**3)) * dt
    return x, y, z

def chenlee(t, a=5, b=-10, c=-0.38):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0, 0
    for i in range(1, t.size):
        x[i] = x[i-1] + (a * x[i-1] - y[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (b * y[i-1] + x[i-1] * z[i-1]) * dt
        z[i] = z[i-1] + (c * z[i-1] + x[i-1] * y[i-1] / 3) * dt
    return x, y, z

def lorenz_mod2(t, alpha=0.9, beta=5, gamma=9.9):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (-alpha * x[i-1] + y[i-1] * y[i-1] - z[i-1] * z[i-1] + alpha * gamma) * dt
        y[i] = y[i-1] + (x[i-1] * (y[i-1] - beta * z[i-1])) * dt
        z[i] = z[i-1] + (-z[i-1] + x[i-1] * y[i-1]) * dt
    return x, y, z

def dadras(t, a=3, b=2.7, c=1.7, d=2, e=9):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (y[i-1] - a * x[i-1] + b * y[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (c * y[i-1] - x[i-1] * z[i-1] + z[i-1]) * dt
        z[i] = z[i-1] + (d * x[i-1] * y[i-1] - e * z[i-1]) * dt
    return x, y, z

def halvorsen(t, a=1.4):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (-a * x[i-1] - 4 * y[i-1] - 4 * z[i-1] - y[i-1] * y[i-1]) * dt
        y[i] = y[i-1] + (-a * y[i-1] - 4 * z[i-1] - 4 * x[i-1] - z[i-1] * z[i-1]) * dt
        z[i] = z[i-1] + (-a * z[i-1] - 4 * x[i-1] - 4 * y[i-1] - x[i-1] * x[i-1]) * dt
    return x, y, z

def hadley(t, alpha=0.2, beta=4, delta=8):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (-alpha * x[i-1] + y[i-1] * y[i-1] - z[i-1] * z[i-1] + alpha * delta) * dt
        y[i] = y[i-1] + (x[i-1] * (y[i-1] - beta * z[i-1])) * dt
        z[i] = z[i-1] + (-z[i-1] + x[i-1] * y[i-1]) * dt
    return x, y, z

def lu(t, a=36, b=3, c=20):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (a * (y[i-1] - x[i-1])) * dt
        y[i] = y[i-1] + (c * x[i-1] - x[i-1] * z[i-1] + c * y[i-1]) * dt
        z[i] = z[i-1] + (x[i-1] * y[i-1] - b * z[i-1]) * dt
    return x, y, z

def newton_leipnik(t, a=0.4, b=0.175):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (a * x[i-1] - y[i-1] - 10 * z[i-1] - y[i-1] * y[i-1]) * dt
        y[i] = y[i-1] + (a * y[i-1] + x[i-1] - 5 * z[i-1] - x[i-1] * x[i-1]) * dt
        z[i] = z[i-1] + (b * z[i-1] + x[i-1] * y[i-1] - x[i-1] * z[i-1]) * dt
    return x, y, z

def rikitake(t, mu=2, nu=0.1):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (mu * x[i-1] - nu * y[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (mu * y[i-1] - nu * x[i-1] * z[i-1]) * dt
        z[i] = z[i-1] + (-z[i-1] + x[i-1] * y[i-1]) * dt
    return x, y, z

def sprott(t, a=2.07):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (y[i-1] + a * x[i-1] - x[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (-x[i-1] - y[i-1] * z[i-1]) * dt
        z[i] = z[i-1] + (1 - x[i-1] * y[i-1]) * dt
    return x, y, z

def genesio_tesi(t, a=1.2, b=2.92, c=5):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + y[i-1] * dt
        y[i] = y[i-1] + z[i-1] * dt
        z[i] = z[i-1] + (-a * x[i-1] - b * y[i-1] - c * z[i-1] + x[i-1]**2) * dt
    return x, y, z

def rabinovich_fabrikant(t, alpha=0.1, gamma=0.87):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (y[i-1] * (z[i-1] - 1 + x[i-1]**2) + gamma * x[i-1]) * dt
        y[i] = y[i-1] + (x[i-1] * (3 * z[i-1] + 1 - x[i-1]**2) + gamma * y[i-1]) * dt
        z[i] = z[i-1] + (-2 * z[i-1] * (alpha + x[i-1] * y[i-1])) * dt
    return x, y, z

def bouali(t, alpha=0.3, beta=0.7):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (x[i-1] * (4 - y[i-1]) + alpha * z[i-1]) * dt
        y[i] = y[i-1] + (-y[i-1] * (1 - x[i-1]**2)) * dt
        z[i] = z[i-1] + (-x[i-1] * (beta + z[i-1])) * dt
    return x, y, z

def burke_shaw(t, alpha=10):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (-alpha * x[i-1] + y[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (-y[i-1] + x[i-1] * (z[i-1] + alpha)) * dt
        z[i] = z[i-1] + (1 - x[i-1] * y[i-1]) * dt
    return x, y, z

def coullet(t, a=0.2, b=0.4, c=-0.1):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (x[i-1] * (1 - x[i-1]) - a * y[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (y[i-1] * (1 - y[i-1]) - b * z[i-1] * x[i-1]) * dt
        z[i] = z[i-1] + (z[i-1] * (1 - z[i-1]) - c * x[i-1] * y[i-1]) * dt
    return x, y, z

def dequan_li(t, a=40, b=1.833, c=0.16, d=0.65):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (a * (y[i-1] - x[i-1]) + b * x[i-1] * z[i-1]) * dt
        y[i] = y[i-1] + (d * y[i-1] - x[i-1] * z[i-1]) * dt
        z[i] = z[i-1] + (c * z[i-1] + x[i-1] * y[i-1]) * dt
    return x, y, z

def lotka_volterra(t, alpha=1.5, beta=1, delta=1, gamma=3):
    dt = t[1] - t[0]
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    x[0], y[0], z[0] = 0.1, 0.1, 0.1
    for i in range(1, t.size):
        x[i] = x[i-1] + (alpha * x[i-1] - beta * x[i-1] * y[i-1]) * dt
        y[i] = y[i-1] + (-gamma * y[i-1] + delta * x[i-1] * y[i-1]) * dt
        z[i] = z[i-1] + (-z[i-1] + x[i-1] * y[i-1]) * dt
    return x, y, z

class AttractorPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_attractor(self, func):
        try:
            print(f"Plotting function: {func.__name__}")
            self.ax.clear()
            t = np.linspace(0, 50, 10000)
            x, y, z = func(t)
            self.ax.plot(x, y, z)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred while plotting: {e}")
            print(f"Error in plot_attractor: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Attractor Visualizer')
        self.setGeometry(100, 100, 1000, 600)

        self.canvas = AttractorPlotCanvas(self, width=8, height=6)
        self.load_button = QPushButton('Load Attractors')
        self.load_button.clicked.connect(self.load_attractors)

        self.attractor_list = QListWidget()
        self.attractor_list.clicked.connect(self.plot_selected_attractor)

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.load_button)
        layout.addLayout(left_layout)
        layout.addWidget(self.attractor_list)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.attractors = {
            'lorenz': lorenz,
            'rossler': rossler,
            'thomas': thomas,
            'aizawa': aizawa,
            'chenlee': chenlee,
            'lorenz_mod2': lorenz_mod2,
            'dadras': dadras,
            'halvorsen': halvorsen,
            'hadley': hadley,
            'lu': lu,
            'newton_leipnik': newton_leipnik,
            'rikitake': rikitake,
            'sprott': sprott,
            'genesio_tesi': genesio_tesi,
            'rabinovich_fabrikant': rabinovich_fabrikant,
            'bouali': bouali,
            'burke_shaw': burke_shaw,
            'coullet': coullet,
            'dequan_li': dequan_li,
            'lotka_volterra': lotka_volterra
        }
        self.load_attractors()

    def load_attractors(self):
        try:
            self.attractor_list.clear()
            for name in self.attractors:
                self.attractor_list.addItem(name)
                print(f"Loaded function: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"An error occurred while loading attractors: {e}")
            print(f"Error in load_attractors: {e}")

    def plot_selected_attractor(self):
        try:
            selected_item = self.attractor_list.currentItem()
            if selected_item:
                func_name = selected_item.text()
                func = self.attractors[func_name]
                self.canvas.plot_attractor(func)
                print(f"Selected function: {func_name}")
        except Exception as e:
            QMessageBox.critical(self, "Selection Error", f"An error occurred while plotting the selected attractor: {e}")
            print(f"Error in plot_selected_attractor: {e}")

def main():
    try:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main()
