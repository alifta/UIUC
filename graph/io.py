import os

# Project folder
ROOT = os.path.dirname(os.getcwd())
# Package folder
PACKAGE = os.path.join(ROOT, 'UIUC')
# Data folder
DATA = os.path.join(PACKAGE, 'data')
# Dataset folder
DATASET = os.path.join(PACKAGE, DATA, 'dataset')
# Database folder
DB = os.path.join(PACKAGE, DATA, 'db')
# Extra files folder
FILE = os.path.join(PACKAGE, DATA, 'file')
# Image folder
IMAGE = os.path.join(PACKAGE, DATA, 'image')
# Graph folder
NETWORK = os.path.join(PACKAGE, DATA, 'network')
# HITS score folder
HITS = os.path.join(PACKAGE, DATA, 'hits')


def folder_create():
    """
    Create required folders for the project
    where can be called in __init__.py
    """
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs(DB, exist_ok=True)
    os.makedirs(FILE, exist_ok=True)
    os.makedirs(IMAGE, exist_ok=True)
    os.makedirs(NETWORK, exist_ok=True)
    os.makedirs(HITS, exist_ok=True)