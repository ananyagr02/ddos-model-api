#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies required for building pandas and other scientific packages
apt-get update && apt-get install -y build-essential

# Run the standard pip install
pip install -r requirements.txt