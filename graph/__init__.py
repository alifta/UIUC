import os

# Create the following folders in the project

# Package folder
PACK_PATH = os.getcwd()

# Root folder of project
ROOT_PATH = os.getcwd()

DATA_PATH = os.path.join(ROOT_PATH, 'data')
os.makedirs(DATA_PATH, exist_ok=True)
IMAGE_PATH = os.path.join(ROOT_PATH, 'image')
os.makedirs(IMAGE_PATH, exist_ok=True)
# --- Project Specific Folders ---
DB_PATH = os.path.join(ROOT_PATH, 'db')
os.makedirs(DB_PATH, exist_ok=True)
FILE_PATH = os.path.join(ROOT_PATH, 'file')
os.makedirs(DB_PATH, exist_ok=True)
NET_PATH = os.path.join(ROOT_PATH, 'network')
os.makedirs(NET_PATH, exist_ok=True)
HITS_PATH = os.path.join(ROOT_PATH, 'hits')
os.makedirs(HITS_PATH, exist_ok=True)