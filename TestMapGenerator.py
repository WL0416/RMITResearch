import numpy as np
from PIL import Image
from random import *
from scipy.io import savemat

OUTPUT_FOLDER = ".\\Maps4Test\\"
IMAGE_ROW = 2
IMAGE_COLUMN = 2
IMAGE_CHANNEL = 3
IMAGE_QUANTITY = 200
RANDOM_MINI = 0
RANDOM_MAX = 255


def image_generator():

    image_size = (IMAGE_ROW, IMAGE_COLUMN, IMAGE_CHANNEL)

    tiny_image = np.zeros(image_size, dtype=int)

    print tiny_image.shape

    image_group = []

    image_color_number = IMAGE_ROW * IMAGE_COLUMN * IMAGE_CHANNEL

    while len(image_group) < IMAGE_QUANTITY:

        print len(image_group)

        for index in range(image_color_number):

            color = randint(RANDOM_MINI, RANDOM_MAX)

            dimension_x = int(index / IMAGE_CHANNEL / IMAGE_COLUMN)
            dimension_y = index % IMAGE_COLUMN
            dimension_z = index % IMAGE_CHANNEL

            tiny_image[dimension_x][dimension_y][dimension_z] = color

        image_group.append(tiny_image)

        image = Image.fromarray(tiny_image, 'RGB')

        image.save(OUTPUT_FOLDER + str(len(image_group)) + '.jpg')

        savemat(OUTPUT_FOLDER + str(len(image_group)) + '.mat', {"Test Image": tiny_image})


def ground_truth():

    return 0


def main():

    image_generator()


if __name__ == "__main__":
    main()