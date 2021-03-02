import os

# Create a list of folders in the project path
# Their name can be used throughout project

# Package folder
PACK_PATH = os.getcwd()

# Root folder of project
ROOT_PATH = os.path.dirname(PACK_PATH)

# Data folder
DATA_PATH = os.path.join(ROOT_PATH, 'data')
os.makedirs(DATA_PATH, exist_ok=True)

# Database folder
DB_PATH = os.path.join(ROOT_PATH, 'db')
os.makedirs(DB_PATH, exist_ok=True)

# Image folder
IMAGE_PATH = os.path.join(ROOT_PATH, 'image')
os.makedirs(IMAGE_PATH, exist_ok=True)

# Extra files folder
FILE_PATH = os.path.join(ROOT_PATH, 'file')
os.makedirs(DB_PATH, exist_ok=True)

# Graph folder
NET_PATH = os.path.join(ROOT_PATH, 'network')
os.makedirs(NET_PATH, exist_ok=True)

# HITS score folder
HITS_PATH = os.path.join(ROOT_PATH, 'hits')
os.makedirs(HITS_PATH, exist_ok=True)