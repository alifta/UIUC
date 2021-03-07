import os

 # Package folder
PACKAGE = os.getcwd()
# Project folder
ROOT = os.path.dirname(PACKAGE)
# Data folder
DATA = os.path.join(ROOT, 'data')
# Dataset folder
DATASET = os.path.join(ROOT, DATA, 'dataset')
# Database folder
DB = os.path.join(ROOT, DATA, 'db')
# Extra files folder
FILE = os.path.join(ROOT, DATA, 'file')
# Image folder
IMAGE = os.path.join(ROOT, DATA, 'image')
# Graph folder
NETWORK = os.path.join(ROOT, DATA, 'network')
# HITS score folder
HITS = os.path.join(ROOT, DATA, 'hits')

def folder_create():
    """
    Create required folders for the project
    where can be called in __init__.py
    """
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(DB, exist_ok=True)
    os.makedirs(DB, exist_ok=True)
    os.makedirs(IMAGE, exist_ok=True)
    os.makedirs(NETWORK, exist_ok=True)
    os.makedirs(HITS, exist_ok=True)