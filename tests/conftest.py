import os
import sys

# Lets the tests import the project modules (phishing/, app.py, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
