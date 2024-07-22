# Download image
! gdown '1i9dqan21DjQoG5Q_VEvm0LrVwAlXD0vB'

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('/content/dog.jpeg')

# Convert the image to grayscale using the Lightness method
gray_img_01 = (img.max(axis=2) + img.min(axis=2)) / 2
gray_img_01[0, 0]

#Convert the img to grayscale using the Average method
gray_img_02 = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
gray_img_02[0, 0]

# Convert the img to grayscale using the Luminosity method
gray_img_03 = (0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2])
gray_img_03[0, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display each grayscale image in a subplot
axes[0].imshow(gray_img_01, cmap='gray')
axes[0].set_title('Lightness Method')

axes[1].imshow(gray_img_02, cmap='gray')
axes[1].set_title('Average Method')

axes[2].imshow(gray_img_03, cmap='gray')
axes[2].set_title('Luminosity Method')

# Hide axes ticks
for ax in axes:
    ax.axis('off')

# Display all figures
plt.show()
