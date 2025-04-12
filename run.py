#!/usr/bin/env python3
"""
LitPipe Web Interface Runner

This script runs the LitPipe web interface.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the main function from the web interface
from web.litpipe_web import main

if __name__ == "__main__":
    main()
