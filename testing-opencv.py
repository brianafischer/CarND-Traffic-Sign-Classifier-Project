import cv2, os, glob
from matplotlib import pyplot as plt

print("Current working directory: ", os.getcwd())
print("")

fig, axes = plt.subplots(1, 5, figsize=(20, 20))
fig.subplots_adjust(hspace=0.05, wspace=0.01)
axes = axes.ravel()

filenames = glob.glob('./web_test_images/*.jpg')
print(filenames)
for i, filename in enumerate(filenames):
    # ToDo: Get traffic sign class (number) of the image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is not None: # OpenCV does not throw an error if the filename is invalid and the read fails!!!
        axes[i].axis('off')
        axes[i].set_title(filename)
        axes[i].imshow(img, cmap='gray')