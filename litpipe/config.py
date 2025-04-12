"""
Configuration for LitPipe - Compatibility Module

This file imports all configuration from the main config.py file.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the main config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import everything from the main config
from config import *

# If any additional configuration is needed specifically for the litpipe module,
# it can be added here.