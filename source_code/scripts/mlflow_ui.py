#!/usr/bin/env python3
"""
Start MLflow UI server with automatic process management.
Cross-platform (Windows, Mac, Linux).

Usage:
    python scripts/mlflow_ui.py                 # Start on default port 5000
    python scripts/mlflow_ui.py 5001           # Start on custom port
"""

import sys
import subprocess
import socket
import time
import os
import signal
import platform


def is_port_in_use(port):
    """Check if port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def kill_process_on_port(port):
    """Kill process using the specified port."""
    if platform.system() == "Windows":
        # `netstat -ano` output ends each matching line with the owning PID;
        # parse it out rather than shelling out with a placeholder PID.
        output = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, check=False
        ).stdout
        pids = {line.split()[-1] for line in output.splitlines() if f":{port} " in line}
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
    else:
        os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")


def main():
    """Start MLflow UI server."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print("=" * 50)
    print("MLflow Server Startup")
    print("=" * 50)
    print()

    # Check if port is already in use
    if is_port_in_use(port):
        print(f"Port {port} is already in use.")
        print()
        print("Options:")
        print("  1. Kill existing process and restart")
        print("  2. Use different port")
        print("  3. Cancel")
        print()

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            print(f"Killing process on port {port}...")
            kill_process_on_port(port)
            time.sleep(2)
            print("Process killed. Starting MLflow...")

        elif choice == "2":
            new_port_str = input("Enter new port (default 5001): ").strip()
            port = int(new_port_str) if new_port_str else 5001
            print(f"Using port {port} instead...")

        elif choice == "3":
            print("Cancelled.")
            sys.exit(0)

        else:
            print("Invalid choice.")
            sys.exit(1)

    print()
    print(f"Starting MLflow on port {port}...")
    print("=" * 50)
    print()

    try:
        subprocess.run(
            ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", str(port)],
            check=False
        )
    except KeyboardInterrupt:
        print()
        print("=" * 50)
        print("MLflow stopped")
        print("=" * 50)
    except Exception as e:
        print(f"Error starting MLflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
