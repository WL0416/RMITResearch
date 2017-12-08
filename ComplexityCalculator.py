from pyheatmap.heatmap import HeatMap
import numpy as np
import cv2 as cv
import math
import os
import xlrd
import time
import progressbar
from scipy.io import savemat

# This is for generating the heat-map of salience map and complexity map

complexity_path = ".\\imageComp\\"
output_path = ".\\heatmaps\\"
file = ".\\Data\\NewData.xls"
images_path = ".\\inputImages\\"
overlay_path = ".\\overlay\\"
combined_path = ".\\combine\\"

# use them in the complexity map generation
THRESHOLD = 230
W_SIZE = 64
W_STEP = 16


def open_excel(access_file):
    try:
        data = xlrd.open_workbook(access_file)
        return data
    except Exception as e:
        print(str(e))


def excel_table_index(access_file):
    data = open_excel(access_file)
    table = data.sheet_by_name("NewData")
    n_rows = table.nrows
    resolutions = []
    resolution = []

    map_data = []
    media_names = []
    gaze_event_duration = []
    gaze_points_x = []
    gaze_points_y = []
    for row in range(1, n_rows-1):
        gpx = table.cell(row, 7).value
        event_type = table.cell(row, 4).value
        if gpx and event_type == 'Fixation':
            resolution.append(int(table.cell(row, 1).value))
            resolution.append(int(table.cell(row, 2).value))
            resolutions.append(resolution)
            resolution = []
            gaze_points_x.append(int(gpx))
            gaze_points_y.append(int(table.cell(row, 8).value))
            media_names.append(table.cell(row, 0).value)
            gaze_event_duration.append(int(table.cell(row, 5).value))

    map_data.append(media_names)
    map_data.append(resolutions)
    map_data.append(gaze_points_x)
    map_data.append(gaze_points_y)
    map_data.append(gaze_event_duration)

    return map_data


# based on fixation data
def heat_map(combine_data):
    map_data = excel_table_index(file)
    media_names = map_data[0]
    resolution = map_data[1]
    gaze_points_x = map_data[2]
    gaze_points_y = map_data[3]
    # GazeEventDuration = map_data[4]

    check_name = media_names[0]

    image_size = [[0, 0], [resolution[0][0], resolution[0][1]]]

    index = 0
    point = []
    pointslist = []
    for name in media_names:
        if check_name is not name:
            print(check_name)
            pointslist.extend(image_size)
            image_size.remove(image_size[1])
            image_size.append([resolution[index][0], resolution[index][1]])

            if combine_data:
                for cindex in range(len(combine_data[0])):
                    if combine_data[0][cindex] == check_name:
                        pointslist.append([int(combine_data[1][cindex]), int(combine_data[2][cindex])])

            hm = HeatMap(pointslist)
            hm_names = check_name.split(".")
            hm_name = hm_names[0]
            # hm.clickmap(save_as=outputpath + hm_name + "_hit.png")
            hm.heatmap(save_as=output_path + hm_name + "_heat.png")
            pointslist = []
            check_name = name

        point.append(gaze_points_x[index])
        point.append(gaze_points_y[index])
        pointslist.append(point)
        point = []

        if index == len(media_names) - 1:
            print(check_name)
            pointslist.extend(image_size)

            if combine_data:
                for cindex in range(len(combine_data[0])):
                    if combine_data[0][cindex] == check_name:
                        pointslist.append([int(combine_data[1][cindex]), int(combine_data[2][cindex])])

            hm = HeatMap(pointslist)
            hm_names = name.split(".")
            hm_name = hm_names[0]
            # hm.clickmap(save_as=outputpath + hm_name + "_hit.png")
            hm.heatmap(save_as=output_path + hm_name + "_heat.png")
            break

        index += 1


# overlap two images together
def overlay(output_image_path, ratio_1, ratio_2):
    for root, folders, files in os.walk(images_path):
        # loop the images
        for each_file in files:
            image = cv.imread(root + "\\" + each_file)
            image_names = each_file.split(".")
            image_name = image_names[0]
            h_map = cv.imread(output_image_path + image_name + "_heat_color.png")

            height, width, depth = image.shape
            print(height, width, depth)

            height, width, depth = h_map.shape
            print(height, width, depth)

            image_map = cv.addWeighted(h_map, ratio_1, image, ratio_2, 0)
            cv.imwrite(overlay_path + image_name + "_complex.jpg", image_map)


# this function is used to combine some data file together.
def combine():
    combined_data = []
    name = []
    gaze_x = []
    gaze_y = []
    for root, folders, files in os.walk(combined_path):
        if files:
            for each_file in files:
                print(each_file)
                data = open_excel(root + "\\" + each_file)
                table = data.sheet_by_name("Data")
                n_rows = table.nrows
                for row in range(1, n_rows - 1):
                    if table.cell(row, 10).value:
                        if table.cell(row, 43).value == 'Fixation':
                            if table.cell(row, 54).value and table.cell(row, 55).value:
                                gaze_x.append(table.cell(row, 54).value)
                                gaze_y.append(table.cell(row, 55).value)
                                name.append(table.cell(row, 10).value)
    combined_data.append(name)
    combined_data.append(gaze_x)
    combined_data.append(gaze_y)
    return combined_data


# calculate the range to data, if the image is grayscale, the range is 0 -> 255
# def range_bytes():
#   return range(256)


# Entropy calculation function, based on Shannon Information Theory
def entropy_calculation(cutout_image):

    if not cutout_image:
        raise ValueError('The data set cannot be empty.')
    entropy = 0
    # for testing all entropy function
    # cutout_image = [0, 0, 1, 1, 2, 2, 3, 3]
    color_list = set(cutout_image)
    # print(len(color_list))

    for element in color_list:

        p_x = float(cutout_image.count(element)) / len(cutout_image)

        if p_x > 0:

            current = - p_x * math.log(p_x, 2)

            # normalised version
            # current = - p_x*math.log(p_x, 2) /math.log(len(color_list),2)

            # this variable entropy represents whole block's entropy value
            entropy += current
            # print(entropy)

    return entropy


# function to do the edge detection by Canny
def canny_edge_detection(cutout_image, window_length, window_wide):

    if not cutout_image:
        raise ValueError('The data set cannot be empty.')

    image = np.array(cutout_image).reshape(window_wide, window_length)

    canny_image = cv.Canny(image, 100, 200)

    return canny_image


# function to do the edge detection by Sobel x
def sobel_edge_detection_x(cutout_image, window_length, window_wide):

    if not cutout_image:
        raise ValueError('The data set cannot be empty.')

    image = np.array(cutout_image).reshape(window_wide, window_length)

    sobel_x_image = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)

    return sobel_x_image


# function to do the edge detection by Sobel y
def sobel_edge_detection_y(cutout_image, window_length, window_wide):

    if not cutout_image:
        raise ValueError('The data set cannot be empty.')

    image = np.array(cutout_image).reshape(window_wide, window_length)

    sobel_y_image = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)

    return sobel_y_image


# function to do the edge detection by Laplacian
def laplacian_edge_detection(cutout_image, window_length, window_wide):

    if not cutout_image:
        raise ValueError('The data set cannot be empty.')

    image = np.array(cutout_image).reshape(window_wide, window_length)

    laplacian_image = cv.Laplacian(image, cv.CV_64F)

    return laplacian_image


'''
def entropy_edge_detection(edge_detection_map):
    pixel_list = edge_detection_map.tolist()
    new_list = generate_new_list(pixel_list)
    entropy = entropy_calculation(new_list)
    return entropy
'''


def generate_new_list(pixel_list):

    new_list = []

    for row in range(len(pixel_list)):

        for column in range(len(pixel_list[row])):

            new_list.append(pixel_list[row][column])

    return new_list


# image complexity heat-map base on Entropy
def calculate_complexity(method="Entropy", window_size=W_SIZE, window_move_step=W_STEP):

    progress = progressbar.ProgressBar()

    # loop the target folder to get the original images' information
    for root, folders, files in os.walk(images_path):
        # loop the images
        for each_file in files:
            print(each_file)

            image = cv.imread(root + "\\" + each_file)
            # print(image)
            # get the images row and column numbers
            row_num = len(image)
            column_num = len(image[0])

            # image resolution
            print(str(column_num) + "*" + str(row_num))

            # list to store the cutout_image used to calculate the complexity_value value
            size = (row_num, column_num)
            final_complexity_map = np.zeros(size)
            counter_map = np.zeros(size, dtype=int)

            # print(final_complexity_map)
            # print (counter_map)

            progress.start(row_num)

            # traverse the matrix
            for row in range(row_num):
                # window moves following y direction
                progress.update(row + 1)

                if row % window_move_step == 0 and (row + window_size) <= row_num:

                    for column in range(column_num):

                        # window moves following x direction
                        if column % window_move_step == 0 and (column + window_size) <= column_num:

                            # the cutout_image represents the pixels in the window
                            cutout_image = []

                            # loop each pixel in the window, color collection
                            for window_column in range(window_size):

                                # here calculate the window pixel's actual location in the final matrix
                                current_column = column + window_column

                                # print('x: '+ str(length))
                                for window_row in range(window_size):
                                    # here calculate the window pixel's actual location in the final matrix
                                    current_row = row + window_row
                                    # print('y: '+ str(wide))
                                    color = image[current_row][current_column][0]
                                    cutout_image.append(color)
                            # print(cutout_image)
                            # calculate the complexity_value value for current window
                            try:
                                if method == "Laplacian":
                                    complexity_value = laplacian_edge_detection(cutout_image,
                                                                                window_size,
                                                                                window_size)

                                elif method == "Sobel_x":
                                    complexity_value = sobel_edge_detection_x(cutout_image,
                                                                              window_size,
                                                                              window_size)

                                elif method == "Sobel_y":
                                    complexity_value = sobel_edge_detection_y(cutout_image,
                                                                              window_size,
                                                                              window_size)

                                elif method == "Canny":
                                    complexity_value = canny_edge_detection(cutout_image,
                                                                            window_size,
                                                                            window_size)
                                else:
                                    complexity_value = entropy_calculation(cutout_image)

                                # here assign complexity_value to each pixel in the window
                                # traverse whole window, the process is the same with color collection
                                for window_column in range(window_size):

                                    current_column = column + window_column

                                    for window_row in range(window_size):
                                        current_row = row + window_row

                                        final_complexity_map[current_row][current_column] += complexity_value
                                        counter_map[current_row][current_column] += 1

                            except ValueError as error:

                                print(repr(error))

            progress.finish()

            final_complexity_map = final_map_generation(row_num, column_num, final_complexity_map, counter_map)

            # print(final_complexity_map)
            # print(counter_map)

            file_names = each_file.split(".")

            filename = file_names[0]

            # output the grayscale image first
            cv.imwrite(complexity_path + filename + "_heat_gray.png", final_complexity_map)
            out_image = cv.imread(complexity_path + filename + "_heat_gray.png")

            # read the grayscale image and make it to be color map
            # color_final_matrix = cv.applyColorMap(out_image, cv.COLORMAP_JET)
            # cv.imwrite(complexity_path + filename + "_heat_color.png", color_final_matrix)

            # save the image as .mat file
            # savemat(complexity_path + filename + '.mat', {"Heat_Map": final_complexity_map})


# traverse final com assign the each element final value which equals to
# sum complexity_value divide the frequency of calculation for each pixel
def final_map_generation(rows, columns, final_map, counter_map):

    for row in range(rows):

        for column in range(columns):

            if counter_map[row][column] == 0:

                counter_map[row][column] = 1

            final_map[row][column] = final_map[row][column] / counter_map[row][column]

    greatest_element = np.amax(np.array(final_map))

    dividend = greatest_element / 255

    # print(dividend)

    for row in range(rows):

        for column in range(columns):

            final_map[row][column] = final_map[row][column] / dividend
            if final_map[row][column] < THRESHOLD:
                final_map[row][column] = 0
            # print(final_map[row][column])

    return final_map


def main():
    # combine_data = combine()
    # generate heat maps of fixation map
    # heat_map(combine_data)
    calculate_complexity()
    # calculate_complexity("Laplacian")
    # calculate_complexity("Sobel_x")
    # calculate_complexity("Sobel_y")
    # calculate_complexity("Canny")
    # overlay(complexity_path, 0.5, 0.5)


if __name__ == "__main__":
    main()