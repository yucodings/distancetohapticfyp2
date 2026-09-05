import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from ui_components import ZoneCard


class UiComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_zone_card_contains_only_name_and_metre_value(self):
        card = ZoneCard("left")
        labels = card.findChildren(QLabel)
        self.assertEqual(len(labels), 2)
        self.assertEqual([label.text() for label in labels], ["LEFT", "-- m"])

        card.set_distance(1.234)

        self.assertEqual([label.text() for label in labels], ["LEFT", "1.23 m"])


if __name__ == "__main__":
    unittest.main()

