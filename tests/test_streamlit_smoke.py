from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.request import urlopen


class StreamlitSmokeTest(unittest.TestCase):
    def test_streamlit_app_serves_localhost(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        port = "8765"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "true",
                "--server.port",
                port,
            ],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.time() + 30
            last_error: Exception | None = None
            while time.time() < deadline:
                try:
                    with urlopen(f"http://localhost:{port}", timeout=2) as response:
                        self.assertEqual(response.status, 200)
                        return
                except Exception as exc:  # pragma: no cover - useful failure detail
                    last_error = exc
                    time.sleep(1)
            self.fail(f"Streamlit did not respond before timeout: {last_error}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
