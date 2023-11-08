import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils import to_categorical

"""
--------------------
    LOAD DATASET
--------------------
"""


# load images
def loadDataImg(img_w, img_h, img_ch, img_data_path):
    """Load images for segmentation

    Args:
        img_w (int): the image weight.
        img_h (int): the image height.
        img_ch (int): the image channels.
        img_data_path (string): path to images directory.

    Returns:
        nd-array: two nd-array with loaded images and masks.
    """
    img_list = sorted(os.listdir(img_data_path))
    images = []
    print(f"Found {len(img_list)} images!")
    print("Reading...")

    i = 0
    for img_name in img_list:
        i += 1
        img = imread(os.path.join(img_data_path, img_name))
        img = np.divide(img.astype(np.float32), 255.0)
        img = resize(img, (img_h, img_w, img_ch), anti_aliasing=True).astype("float32")
        images.append(img)
        # print reading progess
        if i % 10 == 0:
            print(f"{i}/{len(img_list)}")

    return np.array(images)


# load masks
def loadDataMask(img_w, img_h, mask_data_path, binary=True):
    """Load ground truth for segmentation

    Args:
        img_w (int): the image weight.
        img_h (int): the image height.
        img_ch (int): the image channels.
        img_data_path (string): path to images directory.

    Returns:
        nd-array: two nd-array with loaded images and masks.
    """
    mask_list = sorted(os.listdir(mask_data_path))
    masks = []
    print(f"Found {len(mask_list)} masks!")
    print("Reading...")

    i = 0
    for mask_name in mask_list:
        i += 1
        mask = imread(os.path.join(mask_data_path, mask_name))
        if binary:
            mask[mask != 0] = 255.0
            mask = np.divide(mask.astype(np.float32), 255.0)
            mask = resize(mask, (img_h, img_w, 1), order=0, anti_aliasing=False).astype(
                "float32"
            )
        else:
            mask = resize(mask, (img_h, img_w), order=0, anti_aliasing=False).astype(
                "float32"
            )
            unique_labels = np.unique(mask)
            n_classes = 3
            value_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            mask = np.vectorize(lambda x: value_mapping.get(x, x))(mask)
            mask = to_categorical(mask, num_classes=n_classes)
        # print reading progess
        if i % 10 == 0:
            print(f"{i}/{len(mask_list)}")

        masks.append(mask)

    return np.array(masks)


"""
-------------------
    RANDOM CROP
-------------------
"""


# perform random crop
def randomCrop(img, mask, width=512, height=512):
    """Perform random crop on the image and the corresponding mask

    Args:
        img (nparray): the input image.
        mask (nparray): the input mask.
        width (int, optional): the cropped-image width. Defaults to 512.
        height (int, optional): the cropped-image height. Defaults to 512.

    Returns:
        nd-array: two nd-array with cropped image and mask.
    """
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)

    # Perform cropping
    img_cropped = img[y : y + height, x : x + width, :]
    mask_cropped = mask[y : y + height, x : x + width, :]

    # Calculate padding
    pad_top = y
    pad_bottom = img.shape[0] - (y + height)
    pad_left = x
    pad_right = img.shape[1] - (x + width)

    # Pad the areas not kept with zeros (black)
    img_padded = np.pad(
        img_cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )
    mask_padded = np.pad(
        mask_cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )

    return img_padded, mask_padded


# random crop on train and valid set
def get_randomCrop(images, masks, img_w_crop, img_h_crop, img_ch, n_crops):
    """Applies randomCrop() to training and validation set

    Args:
        x_train (nd-array): the input training images.
        y_train (nd-array): the input training masks.
        x_test (nd-array): the input validation images.
        y_test (nd-array): the input validation masks.
        img_w_crop (int): the cropped-image width.
        img_h_crop (int): the cropped-image height.
        img_ch (int): the number of channels.
        n_crops (int): the number of random crops to apply to each image.

    Returns:
        nparray: 4 nd-array with cropped training and validation sets.
    """
    n_img = images.shape[0]
    img_w = images.shape[1]
    img_h = images.shape[2]
    images_crop = np.zeros((n_img * n_crops, img_w, img_h, img_ch))
    masks_crop = np.zeros((n_img * n_crops, img_w, img_h, 3))
    for i in range(n_img):
        for j in range(n_crops):
            (img, msk) = randomCrop(
                images[i, :, :, :], masks[i, :, :, :], img_w_crop, img_h_crop
            )
            images_crop[i * n_crops + j, :, :, :] = img
            masks_crop[i * n_crops + j, :, :, :] = msk
    masks_crop = np.float32(masks_crop)
    return images_crop, masks_crop
