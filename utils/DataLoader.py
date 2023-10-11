import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def gen_labels(im_name, pats):
    """
    Parameters
    ----------
    im_name : Str
    The image file name.
    pat1 : Str
    A string pattern in the filename for 1st class, e.g "Mel"
    pat2 : Str
    A string pattern in the filename 2nd class, e.g, "Nev"

    Returns
    -------
    Label : Numpy array
    Class label of the filename name based on its pattern.
    """
    if len(pats) == 2:
        if pats[0] in im_name:
            label = np.array([0])
        elif pats[1] in im_name:
            label = np.array([1])

    else:
        label = np.array([0] * 9)
        i = 0
        for pat in pats:
            if pat in im_name:
                label[i] = 1
            i += 1

    return label


def get_data(data_path, data_list, img_h, img_w):
    """
    Parameters
    ----------
    data_path : Str
    Path to the data directory
    data_list : List
    A list containing the name of the images.
    img_h : Int
    image height to be resized to.
    img_w : Int
    image width to be resized to.

    Returns
    -------
    img_labels : Nested List
    A nested list containing the loaded images along with their
    corresponding labels.
    """
    img_labels = []
    for item in enumerate(data_list):
        img = imread(os.path.join(data_path, item[1]), as_gray=True)  # "as_grey"
        img = resize(img, (img_h, img_w), anti_aliasing=True).astype("float32")
        if "Skin" in data_path:
            img_labels.append([np.array(img), gen_labels(item[1], ["Mel", "Nev"])])
        elif "Bone" in data_path:
            img_labels.append([np.array(img), gen_labels(item[1], ["AFF", "NFF"])])
        elif "X_ray" in data_path:
            img_labels.append(
                [
                    np.array(img),
                    gen_labels(
                        item[1], ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
                    ),
                ]
            )

        if item[0] % 100 == 0:
            print("Reading: {0}/{1} of train images".format(item[0], len(data_list)))

    shuffle(img_labels)

    return img_labels


def get_data_arrays(nested_list, img_h, img_w):
    """
    Parameters
    ----------
    nested_list : nested list
      nested list of image arrays with corresponding class labels
    img_h : Int
      image height
    img_w : Int
      image width

    Returns
    -------
    img_arrays : Numpy array
      4D Array with the size of (n_data, img_h, img_w, 1)
    label_arrays : Numpy array
      1D array with the size (n_data).
    """
    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype=np.float32)
    label_arrays = np.zeros((len(nested_list), len(nested_list[1][1])), dtype=np.int32)
    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind, :] = nested_list[ind][1]
    img_arrays = np.expand_dims(img_arrays, axis=3)

    return img_arrays, label_arrays


def get_train_test_arrays(
    train_data_path, test_data_path, train_list, test_list, img_h, img_w
):
    """
    Get the directory to the train and test sets, the files names and
    the size of the image and return the image and label arrays for
    train and test sets.
    """
    train_data = get_data(train_data_path, train_list, img_h, img_w)
    test_data = get_data(test_data_path, test_list, img_h, img_w)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del train_data
    del test_data

    return train_img, test_img, train_label, test_label


def loadData(img_w, img_h, rel_path):
    """
    Parameters
    ----------
    img_w : nested list
      Image width
    img_h : Int
      Image height
    rel_path : Str
      Relative path after /home/group_3/Data/

    Returns
    -------
    x_train : Numpy array
    x_test : Numpy array
    y_train : Numpy array
    y_test : Numpy array
    """
    data_path = "/home/group_3/Data/" + rel_path  # Path to data root with two subdirs.

    train_data_path = os.path.join(data_path, "train")
    test_data_path = os.path.join(data_path, "test")
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path, train_list, test_list, img_h, img_w
    )
    return x_train, x_test, y_train, y_test


def get_length(path, subdir):
    """Get the absolute path of a subdirectory

    Args:
        path (string): path of the main directory
        subdir (string): name of the subfolder

    Returns:
        string: absolute path
    """
    length = len(os.listdir(os.path.join(path, subdir)))
    return length


def binary(image):
    image[image != 0] = 255.0
    return image


def generateData(
    training_data_path,
    target_size,
    batch_size,
    class_mode,
    color_mode="rgb",
    classes=None,
    shuffle=True,
    rescale=None,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    validation_split=0.0,
    horizontal_flip=False,
    zoom_range=0.0,
    subset=None,
    seed=None,
    binary=False,
):
    """Generate data using Tensorflow ImageDataGenerator.

    Args:
        scale(int): if None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.
        training_data_path (string): path to the target directory.
        target_size (tuple): list with image width and height as entries.
        batch_size (int): size of the batches of data.
        class_mode (string): one of "categorical", "binary", "sparse", "input", or None.
        color_mode (string): one of "rgb", "grayscale". Default to "rgb
        classes (list): optional list of class subdirectories.
        shuffle (bool): if False, sorts the data in alphanumeric order. Defaults to True.
        validation_split (float): fraction of images reserved for validation (strictly between 0 and 1).
        horizontal_flip (bool): randomly flip inputs horizontally.
        zoom_range (float): range for random zoom.
        subset (string): subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.
        seed (int): optional random seed for shuffling and transformations.
        binary (bool, optional): if True, binarize the loaded masks.


    Returns:
        object: tensorflow data generator.
    """
    if binary:
        preprocessing_function = binary()
    else:
        preprocessing_function = None

    datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        validation_split=validation_split,
        horizontal_flip=horizontal_flip,
        preprocessing_function=preprocessing_function,
    )
    generator = datagen.flow_from_directory(
        training_data_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode=class_mode,
        classes=classes,
        shuffle=shuffle,
        subset=subset,
        seed=seed,
    )

    return generator


def loadDataSeg(img_w, img_h, img_ch, img_data_path, mask_data_path, binary=False):
    """Load data for segmentation (no classes)

    Args:
        img_w (int): the image weight.
        img_h (int): the image height.
        img_ch (int): the image channels.
        img_data_path (string): path to images directory.
        mask_data_path (string): path to masks directory.
        binary (bool, optional): if True, binarize the loaded masks

    Returns:
        nparray: two nparray with loaded images and masks.
    """
    img_list = sorted(os.listdir(img_data_path))
    mask_list = sorted(os.listdir(mask_data_path))

    images = []
    masks = []
    print(f"Found {len(img_list)} images and {len(mask_list)} masks!")
    print("Reading...")

    for img_name, mask_name in zip(img_list, mask_list):
        img = imread(os.path.join(img_data_path, img_name))
        img = np.divide(img.astype(np.float32), 255.0)
        img = resize(img, (img_h, img_w, img_ch), anti_aliasing=True).astype("float32")
        mask = imread(os.path.join(mask_data_path, mask_name))
        if binary:
            mask[mask != 0] = 255.0
        mask = np.divide(mask.astype(np.float32), 255.0)
        mask = resize(mask, (img_h, img_w, 1), anti_aliasing=True).astype("float32")

        images.append(img)
        masks.append(mask)

    print("DONE! :D")
    return np.array(images), np.array(masks)
