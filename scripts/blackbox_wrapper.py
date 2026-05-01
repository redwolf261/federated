#!/usr/bin/env python3
"""Simple cross-platform wrapper to run the Blackbox CLI from Python.

This script prefers a global `blackbox` executable, falls back to `npx -y
blackbox` if available, and prints install instructions otherwise.

Usage examples:
  python scripts/blackbox_wrapper.py --version
  python scripts/blackbox_wrapper.py run "Describe this repo"
"""
import argparse
import shutil
import subprocess
import sys


def find_runner():
    """Return a tuple (runner_cmd, runner_name) or (None, None).

    runner_cmd: list of command tokens to run (prefix)
    runner_name: textual name used in error messages
    """
    if shutil.which("blackbox"):
        return (["blackbox"], "blackbox")
    if shutil.which("npx"):
        return (["npx", "-y", "blackbox"], "npx blackbox")
    return (None, None)


def run_blackbox(argv):
    runner, name = find_runner()
    if not runner:
        print("Blackbox CLI not found in PATH (blackbox or npx).", file=sys.stderr)
        print("Install Node.js (>=20) and then run:", file=sys.stderr)
        print("  npm install -g @blackboxai/cli  # or follow https://github.com/blackboxaicode/cli", file=sys.stderr)
        print("Or use npx: npx -y blackbox", file=sys.stderr)
        return 2

    cmd = runner + argv
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except FileNotFoundError:
        print(f"Runner {name} not executable.", file=sys.stderr)
        return 3


def main():
    parser = argparse.ArgumentParser(description="Run Blackbox CLI with fallback to npx.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to forward to blackbox")
    ns = parser.parse_args()
    argv = ns.args
    if not argv:
        argv = ["--help"]
    rc = run_blackbox(argv)
    sys.exit(rc)


if __name__ == "__main__":
    main()
