import http.server
import os
import shutil
import socketserver
import tempfile
import threading
import unittest

from cognitive_engine.agent.orchestrator import CognitiveEngine


class WebMockHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Square Algorithm Specification</title></head>
        <body>
            <h1>Square Function Reference</h1>
            <p>Verification assertions:</p>
            <code>
                assert square(4) == 16
                assert square(5) == 25
                assert square(2) == 4
            </code>
        </body>
        </html>
        """
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass


class TestWebGroundedVirtualStaged(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_cognitive.db")
        self.server = socketserver.TCPServer(("127.0.0.1", 0), WebMockHandler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_web_grounded_virtual_staged_synthesis(self):
        engine = CognitiveEngine(db_path=self.db_path)
        url = f"http://127.0.0.1:{self.port}/spec"
        goal = f"Synthesize algorithm based on web specification at {url}"

        result = engine.execute_staged_autonomous_cycle(goal)
        self.assertIn("Success", result)

        from skills.default_tenant.square import square
        self.assertEqual(square(4), 16)
        self.assertEqual(square(5), 25)
        self.assertEqual(square(2), 4)


if __name__ == "__main__":
    unittest.main()
