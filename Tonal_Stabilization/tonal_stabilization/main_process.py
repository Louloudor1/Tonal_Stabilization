
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Paramètres de la fenêtre principale ---
        self.setWindowTitle("Tonal Stabilizer")
        self.setGeometry(100, 100, 1000, 600)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())