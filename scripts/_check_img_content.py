import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Dell\Downloads\IMG-20261026-WA0006.jpg")
print("Image shape:", img.shape)
# Let's check mean brightness and whether it has text-like high contrast
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray min:", gray.min(), "max:", gray.max(), "mean:", gray.mean())
# Let's save a smaller preview or check text regions
