Blackbox CLI integration
=========================

This project includes a small Python wrapper at `scripts/blackbox_wrapper.py`
which attempts to run the Blackbox CLI (`blackbox`) if available in PATH and
falls back to `npx -y blackbox` when `npx` is present.

Installation
------------

Blackbox is a Node.js CLI. Install Node.js (>=20) and then either:

- Install globally:

```bash
npm install -g @blackboxai/cli
```

- Or run ad-hoc with `npx`:

```bash
npx -y blackbox
```

Usage
-----

From the repository root:

```bash
# show help
python scripts/blackbox_wrapper.py --help

# run an interactive blackbox session
python scripts/blackbox_wrapper.py

# run a simple CLI command
python scripts/blackbox_wrapper.py --version
```

Notes
-----
- Do not store API keys in the repository. Follow Blackbox CLI docs for
  authentication and configuration (`blackbox configure`).
- The wrapper is intentionally minimal — it only forwards arguments and
  provides user-friendly install guidance when the CLI is not available.
