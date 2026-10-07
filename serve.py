"""Open the web page locally.

    python serve.py              # then open http://localhost:8080
    python serve.py --port 9000

This serves the web/ folder like `python -m http.server`, with one difference:
it tells the browser not to cache anything. Without that, browsers keep old
copies of the CSS and JavaScript for a while, and an edited page can show up
half old and half new.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        # Always send the full file, never "304 Not Modified".
        del self.headers["If-Modified-Since"]
        del self.headers["If-None-Match"]
        super().do_GET()

    def log_message(self, format, *args) -> None:
        pass  # keep the terminal quiet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(NoCacheHandler, directory=str(WEB)))
    except OSError:
        raise SystemExit(f"Port {args.port} is already in use. Try:  python serve.py --port {args.port + 1}")
    print(f"Serving {WEB.name}/ at http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
