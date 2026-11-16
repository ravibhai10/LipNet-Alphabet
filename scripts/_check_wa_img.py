import os
from PIL import Image

path = r"C:\Users\Dell\Downloads\IMG-20261026-WA0006.jpg"
if os.path.exists(path):
    img = Image.open(path)
    print("Format:", img.format, "Size:", img.size, "Mode:", img.mode)
else:
    print("Not found")
