"""CLI health-check script — exits 0 when Supply-Pulse API is healthy."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def check_health(host: str = "localhost", port: int = 8000, timeout: int = 5) -> bool:
    """Return True if the /health endpoint responds with status=ok.

    Args:
        host: API hostname or IP address.
        port: API port number.
        timeout: Request timeout in seconds.

    Returns:
        True if the service is healthy, False otherwise.
    """
    import json

    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
            healthy = data.get("status") == "ok"
            if healthy:
                print(
                    f"[OK] Supply-Pulse healthy — version={data.get('version')} "
                    f"model={data.get('model_version')}"
                )
            else:
                print(f"[WARN] Unexpected health response: {data}")
            return healthy
    except urllib.error.URLError as exc:
        print(f"[ERROR] Could not reach {url}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Health check failed: {exc}")
        return False


def main() -> None:
    """Entry point for the healthcheck CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Supply-Pulse health check")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout seconds (default: 5)")
    args = parser.parse_args()

    healthy = check_health(host=args.host, port=args.port, timeout=args.timeout)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
