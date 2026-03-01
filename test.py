from PIL import Image
import os

directory = "/home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined/images/val"

for file in os.listdir(directory):
    if file.lower().endswith(".jpg"):
        jpg_path = os.path.join(directory, file)
        png_path = os.path.join(directory, file.rsplit(".", 1)[0] + ".png")

        img = Image.open(jpg_path)
        img.save(png_path, "PNG")