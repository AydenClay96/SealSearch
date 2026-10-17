import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class StreamlitAppTests(unittest.TestCase):
    def test_first_reference_image_returns_first_supported_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            class_dir = Path(temporary_directory) / "train" / "Seal_SealID_3"
            class_dir.mkdir(parents=True)
            (class_dir / "z-last.png").touch()
            (class_dir / "a-not-image.txt").touch()
            expected = class_dir / "b-first.jpg"
            expected.touch()

            with patch.object(app, "DATA_DIR", Path(temporary_directory)):
                self.assertEqual(
                    app._first_reference_image("Seal_SealID_3"), expected
                )

    def test_first_reference_image_returns_none_for_missing_class(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(app, "DATA_DIR", Path(temporary_directory)):
                self.assertIsNone(app._first_reference_image("missing"))


if __name__ == "__main__":
    unittest.main()
