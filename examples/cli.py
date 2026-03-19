#!/usr/bin/env python3
"""
Fathom Mode CLI — thin wrapper around ftg.cli.

After installing fathom-mode, you can also run:
    fathom start "Should I buy a car or invest?"
    fathom relay "Should I change jobs?"

This file exists as a convenience for running without installation:
    python3 examples/cli.py start "..."
"""

from ftg.cli import main

if __name__ == "__main__":
    main()
