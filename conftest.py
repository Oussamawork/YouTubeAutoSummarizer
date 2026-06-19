"""Make the repo root importable so tests can `import scraper`, `import summarizer`, etc.

pytest adds the directory containing this conftest.py to sys.path, so placing it at
the repo root lets the test files import the top-level modules directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
