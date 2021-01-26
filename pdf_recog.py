import cv2 as cv
import numpy as np
import math
import pytesseract
import os
import logging
import configparser
from Params import Params

OUTPUT_FILE = 'res_prom.csv'
IMG_PATH = "imgs"
NUMB_OF_PAGES = 21
MIN_LINE_LENGHT = 600

NUMBER_OF_POINTS = 20
MIN_DIST_BETWEEN_POINTS = 15

COL_FOR_RECOG = 5

page_counter = 0


def dist(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def point_of_intersect_polar(theta1: float, rho1: float, theta2: float, rho2: float) -> list:
    p1 = [int(rho1 * math.cos(theta1)), int(rho1 * math.sin(theta1))]
    rho = rho1 / math.cos(np.pi / 4 - theta1)
    p2 = [int(rho * math.cos(np.pi / 4)), int(rho * math.sin(np.pi / 4))]
    p3 = [int(rho2 * math.cos(theta2)), int(rho2 * math.sin(theta2))]
    rho = rho2 / math.cos(np.pi / 4 - theta2)
    p4 = [int(rho * math.cos(np.pi / 4)), int(rho * math.sin(np.pi / 4))]
    return point_of_intersect(p1, p2, p3, p4)


def intersect_polar_lines(theta1: float, rho11: float, rho12: float, theta2: float, rho21: float, rho22: float) -> list:
    return [
        point_of_intersect_polar(theta1, rho11, theta2, rho21),
        point_of_intersect_polar(theta1, rho12, theta2, rho21),
        point_of_intersect_polar(theta1, rho11, theta2, rho22),
        point_of_intersect_polar(theta1, rho12, theta2, rho22)
    ]


def fill_lines_in_polar(in_img: np.ndarray, params: Params) -> np.ndarray:
    count_of_blur = params.count_of_blur
    img_bl = np.copy(in_img)
    while count_of_blur > 0:
        img_bl = cv.medianBlur(img_bl, 3)
        count_of_blur = count_of_blur - 1
    img_bl = cv.threshold(img_bl, params.thr_border, 256, cv.THRESH_BINARY_INV)[1]
    cv.imwrite("imgs\\thr0.jpg", img_bl)

    lines = cv.HoughLines(img_bl, 1, np.pi / 360.0, params.hough_param_gor)

    debug_img(in_img, lines, "hough_search0.jpg")

    i_ = 0
    while i_ < lines.shape[0]:
        if is_line_t(img_bl, lines[i_][0][0], lines[i_][0][1], 3 * params.hough_param_gor // 4) == 0:
            lines = np.delete(lines, i_, axis=0)
        else:
            for j_ in range(i_):
                if not (is_one_theta(lines[i_][0][1], lines[j_][0][1]) and math.fabs(
                        lines[i_][0][0] - lines[j_][0][0]) < params.min_dist_between_points):
                    continue
                lines = np.delete(lines, i_, axis=0)
                break
            else:
                i_ = i_ + 1

    return lines


def debug_img(d_img: np.ndarray, lines: np.ndarray, name: str):
    pr_img = np.copy(d_img)
    for i_ in range(0, len(lines)):
        rho = lines[i_][0][0]
        theta = lines[i_][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv.line(pr_img, pt1, pt2, (0, 0, 255), 20)
    cv.imwrite("imgs\\" + name, pr_img)


def is_line_t(in_img: np.ndarray, rho: float, theta: float, thresh_val: int = 400) -> int:
    if math.fabs(theta - np.pi / 4) < 0.1 or math.fabs(theta - 3 * np.pi / 4) < 0.1 or math.fabs(
            rho) == 0. and math.fabs(theta) == 0.:
        return 0

    thr_img = in_img
    p0 = [rho * math.sin(theta), rho * math.cos(theta)]
    v = [- math.cos(theta), math.sin(theta)]

    if math.fabs(v[0]) > math.fabs(v[1]):
        x_cur = 0
        coeff = v[1] / v[0]
        p1 = [x_cur, round(p0[1] + (x_cur - p0[0]) * coeff)]
        while p1[1] <= 0 or p1[1] > in_img.shape[1] - 1:
            x_cur = x_cur + 20
            p1 = [x_cur, round(p0[1] + (x_cur - p0[0]) * coeff)]
        max_len = 0
        cur_len = 0
        while (0 <= p1[0] < in_img.shape[0]) and (0 < p1[1] < in_img.shape[1]):
            if thr_img[p1[0], p1[1]] == 255:
                cur_len = cur_len + 1
            elif cur_len > 0:
                if cur_len > max_len:
                    max_len = cur_len
                cur_len = 0

            x_cur = x_cur + 1
            p1 = [x_cur, round(p0[1] + (x_cur - p0[0]) * coeff)]
        if cur_len > max_len:
            max_len = cur_len
        if max_len > thresh_val:
            return max_len
        else:
            return 0
    else:
        y_cur = 0
        coeff = v[0] / v[1]
        p1 = [round(p0[0] + (y_cur - p0[1]) * coeff), y_cur]
        while p1[0] <= 0 or p1[0] > in_img.shape[0] - 1:
            y_cur = y_cur + 20
            p1 = [round(p0[0] + (y_cur - p0[1]) * coeff), y_cur]
        max_len = 0
        cur_len = 0
        while (0 <= p1[1] < in_img.shape[1]) and (0 < p1[0] < in_img.shape[0]):
            if thr_img[p1[0], p1[1]] != 0:
                cur_len = cur_len + 1
            elif cur_len > 0:
                if cur_len > max_len:
                    max_len = cur_len
                cur_len = 0

            y_cur = y_cur + 1
            p1 = [round(p0[0] + (y_cur - p0[1]) * coeff), y_cur]
        if cur_len > max_len:
            max_len = cur_len
        if max_len > thresh_val:
            return max_len
        else:
            return 0


def is_one_theta(theta1: float, theta2: float) -> bool:
    return (math.fabs(theta1 - theta2) < 0.1) or (math.fabs(theta1 - np.pi) < 0.1 and math.fabs(theta2) < 0.1) or (
            math.fabs(theta2 - np.pi) < 0.1 and math.fabs(theta1) < 0.1)


def detect_table(in_img: np.ndarray, hough_param: int, min_dist_between_points: int) -> list:
    lines = fill_lines_in_polar(in_img, hough_param, min_dist_between_points)
    # debug_img(in_img, lines, "first_search" + str(page_counter) + ".jpg")
    thetas = []
    for line in lines:
        for i_ in range(len(thetas)):
            if is_one_theta(thetas[i_][0], line[0][1]):
                thetas[i_][1] = thetas[i_][1] + 1
                break
        else:
            thetas.append([line[0][1], 1])

    thetas.sort(key=lambda theta: theta[1])
    theta_1 = thetas[0][0]
    theta_2 = thetas[1][0]
    if theta_1 > theta_2:
        buf = theta_1
        theta_1 = theta_2
        theta_2 = buf

    if is_one_theta(theta_2, 0):
        buf = theta_2 - np.pi
        theta_2 = theta_1
        theta_1 = buf

    rho_1_1 = 100000.
    rho_1_2 = 0.
    rho_2_1 = 100000.
    rho_2_2 = 0.
    for line in lines:
        if is_one_theta(line[0][1], theta_1) and math.fabs(line[0][0]) < rho_1_1:
            rho_1_1 = math.fabs(line[0][0])
        if is_one_theta(line[0][1], theta_1) and math.fabs(line[0][0]) > rho_1_2:
            rho_1_2 = math.fabs(line[0][0])
        if is_one_theta(line[0][1], theta_2) and line[0][0] < rho_2_1:
            rho_2_1 = line[0][0]
        if is_one_theta(line[0][1], theta_2) and line[0][0] > rho_2_2:
            rho_2_2 = line[0][0]

    # print(str(theta_1) + " " + str(rho_1_1) + " " + str(rho_1_2))
    # print(str(theta_2) + " " + str(rho_2_1) + " " + str(rho_2_2))
    return [intersect_polar_lines(theta_1, rho_1_1, rho_1_2, theta_2, rho_2_1, rho_2_2), theta_1]


def rotate_img(table_img: np.ndarray, angle: float) -> np.ndarray:
    image_center = tuple(np.array(table_img.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(table_img.shape[0] * abs_sin + table_img.shape[1] * abs_cos)
    bound_h = int(table_img.shape[0] * abs_cos + table_img.shape[1] * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    return cv.warpAffine(table_img, rot_mat, (bound_w, bound_h), borderValue=(255, 255, 255))


def rotate_table(table_img: np.ndarray, table_corners: list) -> np.ndarray:
    print("Corners:")
    print(table_corners[0])
    print(table_corners[2])
    alpha = math.atan2(table_corners[2][0] - table_corners[0][0], table_corners[2][1] - table_corners[0][1])
    alpha = alpha * 180 / np.pi
    print(-alpha)
    return rotate_img(table_img, -alpha)


def cut_table(in_img: np.ndarray) -> np.ndarray:
    lims, theta = detect_table(in_img)
    print(lims)
    if len(lims) == 0:
        raise Exception("table not found")
    x1 = in_img.shape[1]
    x2 = 0
    y1 = in_img.shape[0]
    y2 = 0

    for corner in lims:
        if corner[0] < x1:
            x1 = corner[0]
        if corner[0] > x2:
            x2 = corner[0]
        if corner[1] < y1:
            y1 = corner[1]
        if corner[1] > y2:
            y2 = corner[1]

    if x1 < 0:
        x1 = 0
    if x2 > in_img.shape[1]:
        x2 = in_img.shape[1]
    if y1 < 0:
        y1 = 0
    if y2 > in_img.shape[0]:
        y2 = in_img.shape[0]

    theta = theta * 180 / np.pi
    return rotate_img(in_img[y1:y2, x1:x2], theta)


def debug_lines(table_img: np.ndarray, gorizontal_lines: list, vertical_lines: list):
    draw_img = np.copy(table_img)
    for line in gorizontal_lines:
        cv.line(draw_img, (0, line[0]), (draw_img.shape[1], line[1]), (0, 0, 255), 10)
    for line in vertical_lines:
        cv.line(draw_img, (line[0], 0), (line[1], draw_img.shape[0]), (0, 0, 255), 10)
    cv.imwrite("imgs\\debug_table_" + str(page_counter) + ".jpg", draw_img)


def coord_vert(max_x: int, rho: float, theta: float) -> list:
    p0 = [rho * math.sin(theta), rho * math.cos(theta)]
    coeff = -math.sin(theta) / math.cos(theta)
    y0 = p0[1] - p0[0] * coeff
    y_l = p0[1] + (max_x - p0[0]) * coeff
    return [round(y0), round(y_l)]


def coord_gor(max_y: int, rho: float, theta: float) -> list:
    p0 = [rho * math.sin(theta), rho * math.cos(theta)]
    coeff = -math.cos(theta) / math.sin(theta)
    x0 = p0[0] - p0[1] * coeff
    x_l = p0[0] + (max_y - p0[1]) * coeff
    return [round(x0), round(x_l)]


def detect_lines(table_img: np.ndarray, params: Params) -> list:
    all_lines = fill_lines_in_polar(table_img, params)
    # debug_img(table_img, all_lines, "second_search" + str(page_counter) + ".jpg")
    gorizontal_lines = []
    vertical_lines = []
    for line in all_lines:
        if is_one_theta(line[0][1], np.pi):
            vertical_lines.append(coord_vert(table_img.shape[0] - 1, line[0][0], line[0][1]))
        else:
            gorizontal_lines.append(coord_gor(table_img.shape[1] - 1, line[0][0], line[0][1]))
    gorizontal_lines.sort(key=lambda gor_line: gor_line[0])
    vertical_lines.sort(key=lambda vert_line: vert_line[0])

    i_ = 1
    while i_ < len(gorizontal_lines):
        if gorizontal_lines[i_][0] - gorizontal_lines[i_ - 1][0] < params.min_dist_between_points:
            del gorizontal_lines[i_]
        else:
            i_ = i_ + 1

    i_ = 1
    while i_ < len(vertical_lines):
        if vertical_lines[i_][0] - vertical_lines[i_ - 1][0] < 40:
            del vertical_lines[i_]
        else:
            i_ = i_ + 1

    """
    if gorizontal_lines[0][0] > 120:
        gorizontal_lines.insert(0, [0, 0])

    if gorizontal_lines[-1][0] < table_img.shape[0] - 120:
        gorizontal_lines.append([table_img.shape[0] - 1, table_img.shape[0] - 1])

    if vertical_lines[0][0] > 50:
        vertical_lines.insert(0, [0, 0])

    if vertical_lines[-1][0] < table_img.shape[1] - 120:
        vertical_lines.append([table_img.shape[1] - 1, table_img.shape[1] - 1])
    """
    res = []
    print(vertical_lines)
    debug_lines(table_img, gorizontal_lines, vertical_lines)
    for i_ in range(len(gorizontal_lines)):
        res.append([])
        for j_ in range(len(vertical_lines)):
            res[-1].append(point_of_intersect(
                [0, vertical_lines[j_][0]],
                [table_img.shape[0] - 1, vertical_lines[j_][1]],
                [gorizontal_lines[i_][0], 0],
                [gorizontal_lines[i_][1], table_img.shape[1] - 1]
            ))
    return res


# type_c: 0 - up, 1 - left, 2 - right, 3 - down
def is_line(cv_img: np.ndarray, i: int, j: int, type_c: int, number_of_points: int, corridor: int) -> bool:
    if type_c == 0:
        for i_ in range(i + 2, i + number_of_points + 2):
            pr_res = False
            for cor in range(min(corridor, min(cv_img.shape[1] - j, j))):
                pr_res = (pr_res or (cv_img[i_, j + cor] == 255) or (cv_img[i_, j - cor] == 255))
            if not pr_res:
                return False
    elif type_c == 1:
        for j_ in range(j + 2, j + number_of_points + 2):
            pr_res = False
            for cor in range(corridor):
                pr_res = (pr_res or (cv_img[i + cor, j_] == 255) or (cv_img[i - cor, j_] == 255))
            if not pr_res:
                return False
    elif type_c == 2:
        for j_ in range(j - 2, j - number_of_points - 2, -1):
            pr_res = False
            for cor in range(corridor):
                pr_res = (pr_res or (cv_img[i + cor, j_] == 255) or (cv_img[i - cor, j_] == 255))
            if not pr_res:
                return False
    elif type_c == 3:
        for i_ in range(i - 2, i - number_of_points - 2, -1):
            pr_res = False
            for cor in range(corridor):
                pr_res = (pr_res or (cv_img[i_, j + cor] == 255) or (cv_img[i_, j - cor] == 255))
            if not pr_res:
                return False
    return True


# type_c: 0 - up, 1 - left, 2 - right, 3 - down
def fill_lines(cv_img: np.ndarray, p1: list, p2: list, type_c: int, params: Params, more_inf=None) -> list:
    if more_inf is None:
        more_inf = []
    res = []
    if type_c == 0 or type_c == 3:
        delta = (p2[0] - p1[0]) / (p2[1] - p1[1])
    else:
        delta = (p2[1] - p1[1]) / (p2[0] - p1[0])

    if type_c == 0:
        for j in range(p1[1] + 10, p2[1] - 10):
            i = math.floor(delta * (j - p1[1])) + p1[0]
            if is_line(cv_img, i, j, type_c, params.count_of_points, params.coridor):
                res.append([i, j])
    elif type_c == 1:
        for i in range(p1[0] + 10, p2[0] - 10):
            j = math.floor(delta * (i - p1[0])) + p1[1]
            if is_line(cv_img, i, j, type_c, params.count_of_points, params.coridor):
                res.append([i, j])
                if len(res) > 2:
                    i += 3 * (res[-1][0] - res[-2][0]) // 4
    elif type_c == 2:
        for found_p in more_inf[1]:
            for i in range(found_p[0] + more_inf[0] - params.min_dist_between_points // 2,
                           found_p[0] + more_inf[0] + params.min_dist_between_points // 2):
                j = math.floor(delta * (i - p1[0])) + p1[1]
                if i < cv_img.shape[0] and j < cv_img.shape[1] and is_line(cv_img, i, j, type_c, params.count_of_points, params.coridor):
                    res.append([i, j])
                    break
            else:
                print('Error: not found right bi-point for ')
                print(found_p)
                print('Use approximate value')
                res.append([found_p[0] + more_inf[0], math.floor(delta * (found_p[0] + more_inf[0] - p1[0])) + p1[1]])
    elif type_c == 3:
        for found_p in more_inf[1]:
            for j in range(found_p[1] + more_inf[0] - params.min_dist_between_points // 2,
                           found_p[1] + more_inf[0] + params.min_dist_between_points // 2):
                i = math.floor(delta * (j - p1[1])) + p1[0]
                if is_line(cv_img, i, j, type_c, params.count_of_points, params.coridor):
                    res.append([i, j])
                    break
            else:
                print('Error: not found down bi-point for ')
                print(found_p)
                print('Use approximate value')
                res.append([math.floor(delta * (found_p[1] + more_inf[0] - p1[1])) + p1[0], found_p[1] + more_inf[0]])

    return res


def clean_points(points: list, min_dist_between_points: int) -> list:
    i = 0
    while i < len(points):
        j = i + 1
        while j < len(points):
            if dist(points[i], points[j]) < min_dist_between_points:
                points.remove(points[j])
            else:
                break
        i += 1
    return points


def intersect_coeff(p1: list, p2: list, p3: list, p4: list) -> float:
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p4[0] - p3[0], p4[1] - p3[1]]

    if v2[0] == 0:
        return (p3[0] - p1[0]) / v1[0]
    delta = v2[1] / v2[0]
    return (p1[1] - p3[1] - delta * (p1[0] - p3[0])) / (delta * v1[0] - v1[1])


def point_of_intersect(p1: list, p2: list, p3: list, p4: list) -> list:
    t = intersect_coeff(p1, p2, p3, p4)
    return [round(p1[0] + t * (p2[0] - p1[0])), round(p1[1] + t * (p2[1] - p1[1]))]


def is_changed_frame(thr: np.ndarray, res: np.ndarray, i_: int, j_: int, is_in: bool) -> int:
    if is_in and thr[i_, j_] == 255:
        return 0
    if thr[i_, j_] == 0:
        res[i_, j_] = 255
        return 1
    return -1


def kill_frame(img: np.ndarray) -> np.ndarray:
    res = img.copy()
    sh = res.shape
    thresh = cv.threshold(res, 220, 255, cv.THRESH_BINARY)[1]
    max_dist = 5  #  sh[0] // 10

    # kill upper path of frame
    for j_ in range(sh[1]):
        is_in_fr = False
        for i_ in range(max_dist):
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 0:
                break
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 1:
                is_in_fr = True

    # kill left path of frame
    for i_ in range(sh[0]):
        is_in_fr = False
        for j_ in range(max_dist):
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 0:
                break
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 1:
                is_in_fr = True

    # kill right path of frame
    for i_ in range(sh[0]):
        is_in_fr = False
        for j_ in range(sh[1] - 1, sh[1] - max_dist, -1):
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 0:
                break
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 1:
                is_in_fr = True

    # kill down path of frame
    for j_ in range(sh[1]):
        is_in_fr = False
        for i_ in range(sh[0] - 1, sh[0] - max_dist, -1):
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 0:
                break
            if is_changed_frame(thresh, res, i_, j_, is_in_fr) == 1:
                is_in_fr = True

    return res


def cutting_img(img: np.ndarray) -> np.ndarray:
    sh = img.shape
    thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)[1]
    left_l = 0
    right_l = sh[1] - 1

    black_found = False
    j = 0
    while (not black_found) and (j < right_l):
        for i in range(sh[0]):
            if thresh[i, j] == 0:
                black_found = True
                if j > 3:
                    left_l = j - 3
                break
        else:
            j += 1

    black_found = False
    j = right_l
    while (not black_found) and (j > left_l):
        for i in range(sh[0]):
            if thresh[i, j] == 0:
                black_found = True
                if j < right_l - 3:
                    right_l = j + 3
                break
        else:
            j -= 1

    return img[:, left_l:right_l]


def increase_img(img: np.ndarray) -> np.ndarray:
    scale_percent = 300
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]

    return cv.resize(img, dim, interpolation=cv.INTER_LINEAR)


def optimize_img(img: np.ndarray) -> np.ndarray:
    cut_img = cutting_img(img)
    big_img = increase_img(cut_img)
    return big_img


def recog_text(img: np.ndarray, p1: list, p2: list, p3: list, p4: list, i: int = -1, j: int = -1, params: Params = None) -> str:
    cf = configparser.ConfigParser()
    cf.read("settings.ini")
    pytesseract.pytesseract.tesseract_cmd = cf.get("paths", "tesseract_path")
    min_i = max((p1[0] + p2[0]) // 2, 0)
    max_i = min((p3[0] + p4[0]) // 2, img.shape[0] - 1)
    min_j = max((p1[1] + p3[1]) // 2, 0)
    max_j = min((p2[1] + p4[1]) // 2, img.shape[1] - 1)

    img_with_text = img[min_i:max_i, min_j:max_j]
    if params.kill_border:
        img_with_text = kill_frame(img_with_text)
    # img_with_text = cv.medianBlur(img_with_text, 3)
    img_with_text = cv.threshold(img_with_text, params.thr_text, 255, cv.THRESH_BINARY)[1]

    # cv.imwrite("imgs\\tess\\inv_n_" + str(page_counter) + "_" + str(i) + ".jpg", img_with_text)

    opt_image = optimize_img(img_with_text)
    cv.imwrite(str(i) + str(j) + ".jpg", opt_image)
    text = pytesseract.image_to_string(opt_image, lang='rus').replace('\n', ' ').replace('\t', ' ')
    if text.strip() == "":
        text = pytesseract.image_to_string(opt_image, config='digits').replace('\n', ' ').replace('\t', ' ')

    return text.replace(".", "").replace(",", "").replace("—", "").strip()


def is_white_point(table_img: np.ndarray, i_coord: int, j_coord: int, count: int) -> bool:
    res = True
    for i_ in range(count):
        res = res and table_img[i_coord + i_, j_coord] == 255
        res = res and table_img[i_coord - i_, j_coord] == 255
    return res


def white_lines(img: np.ndarray) -> list:
    # find upper point
    pos_1 = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 5, 0, -1):
            if img[i, j] == 0 and img[i + 5, j] == 0:
                pos_1 = [j - 20, j + 20]
                pos_2 = i - 5
                break
        if pos_1 != 0:
            break

    # find vertical
    vertical_lines = []
    i = pos_1[0]
    while i > 0:
        for j in range(pos_2, img.shape[0] - 100):
            if img[j, i] != 255 or img[j + 1, i] != 255 or img[j - 1, i] != 255:
                i = i - 1
                break
        else:
            vertical_lines.append(i - 5)
            i = i - 100

    i = pos_1[0] + 100
    while i < img.shape[1]:
        for j in range(pos_2, img.shape[0] - 100):
            if img[j, i] != 255 or img[j + 1, i] != 255 or img[j - 1, i] != 255:
                i = i + 1
                break
        else:
            vertical_lines.append(i + 5)
            i = i + 100
    vertical_lines.sort()
    while len(vertical_lines) > 3 and vertical_lines[2] - vertical_lines[1] < 300:
        del vertical_lines[0]
    vertical_lines.insert(0, 0)
    # for coord in vertical_lines:
    # cv.line(table_1, (coord, 0), (coord, img_1.shape[0]), (0, 0, 0), 10)
    print(vertical_lines)

    # find gorizontal
    gorizontal_lines = [pos_2]
    i = pos_2 + 50
    while i < img.shape[0] - 7:
        for j in range(vertical_lines[1], vertical_lines[2]):
            if not is_white_point(img, i, j, 7):
                i = i + 1
                break
        else:
            gorizontal_lines.append(i + 20 + 20)
            i = i + 100

    if gorizontal_lines[-1] < img.shape[0] - 100:
        gorizontal_lines.append(img.shape[0] - 1 + 20)

    # for coord in gorizontal_lines:
    # cv.line(table_1, (0, coord), (img_1.shape[1], coord), (0, 0, 0), 10)

    # cv.imwrite("imgs\\table_" + str(page_counter) + "_0.jpg", table_1)
    points_of_table = []
    for i in gorizontal_lines:
        points_of_table.append([])
        for j in vertical_lines:
            points_of_table[-1].append([i, j])
    return points_of_table


def is_gor_line(in_img: np.ndarray, right_pos: int) -> int:
    for j_ in range(in_img.shape[1] - 1 - 20, in_img.shape[1] - 1 - 20 - NUMBER_OF_POINTS, -1):
        if in_img[right_pos, j_] != 0:
            return -1

    left_pos = right_pos - 10
    while left_pos < right_pos + 10:
        for j_ in range(20, 20 + NUMBER_OF_POINTS // 2):
            if in_img[left_pos, j_] != 0:
                break
        else:
            return (left_pos + right_pos) // 2
        left_pos = left_pos + 1
    print("Not found left point for " + str(right_pos))
    return right_pos


def image_recog(in_img: np.ndarray, doc: int, page: int) -> str:
    res = ""
    cv_thr = cv.threshold(in_img, 180, 255, cv.THRESH_BINARY)[1]
    cv.imwrite("thr_col_" + str(doc) + "_" + str(page) + ".jpg", cv_thr)
    cur_i = 0
    i = 10
    while i < cv_thr.shape[0] - 20:
        buf = is_gor_line(cv_thr, i)
        if buf == -1:
            i = i + 1
            continue
        res_of_recog = recog_text(in_img,
                                  [cur_i, 0], [cur_i, in_img.shape[1] - 1],
                                  [buf, 0], [buf, in_img.shape[1] - 1])
        res = res + str(doc) + ";" + str(page) + ";" + res_of_recog + ";\n"
        i = i + 10
        cur_i = buf
    res_of_recog = recog_text(in_img,
                              [cur_i, 0], [cur_i, in_img.shape[1] - 1],
                              [in_img.shape[0] - 1, 0], [in_img.shape[0] - 1, in_img.shape[1] - 1])
    return res + str(doc) + ";" + str(page) + ";" + res_of_recog + ";\n"


if __name__ == "__main__":
    with open(OUTPUT_FILE, 'w') as f:
        # bad_pages = [21, 22, 25, 30, 32, 34, 42, 45, 47, 48, 51, 52, 53]
        files_ = [6, 14, 6, 15, 14, 1, 14, 2, 14, 3, 14, 4, 14, 5]
        for i_ in range(0, len(files_), 2):
            name = "imgs\\Acts3\\img" + str(files_[i_]) + "_" + str(files_[i_ + 1]) + ".jpg"
            cv_img = cv.imread(name, cv.IMREAD_GRAYSCALE)
            buf = image_recog(cv_img, files_[i_], files_[i_ + 1])
            f.write(buf)

        while page_counter < NUMB_OF_PAGES:
            page_counter += 1
            # if page_counter > 1:
            #     continue
            print('Page ' + str(page_counter))
            cv_img = cv.imread("imgs\\img" + str(page_counter) + ".jpg", cv.IMREAD_GRAYSCALE)
            # if page_counter != 21:
            #     cv_img = cv.rotate(cv_img, cv.ROTATE_90_CLOCKWISE)
            # else:
            # cv_img = cv.rotate(cv_img, cv.ROTATE_90_COUNTERCLOCKWISE)
            # img_1 = cv.imread("imgs\\img" + str(page_counter) + "_0.jpg", cv.IMREAD_GRAYSCALE)
            # table_1 = np.copy(img_1)
            # img = cv.threshold(img_1[20:img_1.shape[0] - 30, :], 190, 255, cv.THRESH_BINARY)[1]

            # cv_img = cv.rotate(cv_img, cv.ROTATE_90_CLOCKWISE)
            
            try:
                table = cut_table(cv_img)
            except Exception as exc:
                print(exc.args[0])
                print('End of page')
                continue
            
            points_of_table = detect_lines(cv_img)

            down_limit = len(points_of_table) - 1
            right_limit = len(points_of_table[0]) - 1
            # Распознавание текста в каждой ячейке

            # if page_counter == 24 or page_counter == 26:
            #     ind = COL_FOR_RECOG - 1
            # else:
            ind = COL_FOR_RECOG

            corners = [points_of_table[0][ind], points_of_table[0][ind + 1],
                       points_of_table[-1][ind], points_of_table[-1][ind + 1]]

            corns = np.copy(cv_img)
            cv.line(corns, (corners[0][1], corners[0][0]), (corners[1][1], corners[1][0]), (0, 0, 0), 10)
            cv.line(corns, (corners[0][1], corners[0][0]), (corners[2][1], corners[2][0]), (0, 0, 0), 10)
            cv.line(corns, (corners[3][1], corners[3][0]), (corners[1][1], corners[1][0]), (0, 0, 0), 10)
            cv.line(corns, (corners[3][1], corners[3][0]), (corners[2][1], corners[2][0]), (0, 0, 0), 10)
            cv.imwrite("imgs\\table_" + str(page_counter) + ".jpg", corns)

            thresh_ = cv.threshold(cv_img, 220, 255, cv.THRESH_BINARY_INV)[1]
            cv.imwrite("imgs\\thr_" + str(page_counter) + ".jpg", thresh_)
            # Поиск линий таблицы
            upper_lines = []  # clean_points(fill_lines(thresh_, corners[0], corners[1], 0))
            left_lines = clean_points(fill_lines(thresh_, corners[0], corners[2], 1))
            right_lines = fill_lines(thresh_, corners[1], corners[3], 2,
                                     [corners[1][0] - corners[0][0], left_lines])
            down_lines = []  # clean_points(fill_lines(thresh_, corners[2], corners[3], 3,
            #                     [corners[2][1] - corners[0][1], upper_lines]))

            for i_ in range(len(left_lines)):
                cv.line(corns, (left_lines[i_][1], left_lines[i_][0]), (right_lines[i_][1], right_lines[i_][0]),
                        (0, 0, 0),
                        10)
            cv.imwrite("imgs\\table_" + str(page_counter) + ".jpg", corns)

            if len(upper_lines) == len(down_lines):
                print('OK up down')
            else:
                print('Error up down ' + str(len(upper_lines)) + ' ' + str(len(down_lines)))
                break

            if len(left_lines) == len(right_lines):
                print('OK left right')
            else:
                print('Error left right ' + str(len(left_lines)) + ' ' + str(len(right_lines)))
                print(left_lines)
                print(right_lines)
                break

            # Расчёт матрицы координат ячеек таблицы
            points_of_table = [[corners[0]] + upper_lines + [corners[1]]]
            for i in range(len(left_lines)):
                points_of_table.append([left_lines[i]] + [[0, 0]] * len(upper_lines) + [right_lines[i]])
            points_of_table.append([corners[2]] + down_lines + [corners[3]])

            down_limit = len(points_of_table) - 1
            right_limit = len(points_of_table[0]) - 1
            for i in range(1, down_limit):
                for j in range(1, right_limit):
                    points_of_table[i][j] = point_of_intersect(
                        points_of_table[0][j],
                        points_of_table[down_limit][j],
                        points_of_table[i][0],
                        points_of_table[i][right_limit])

            for i in range(down_limit):
                f.write(str(page_counter) + ";")
                for j in range(right_limit):
                    f.write(recog_text(cv_img, points_of_table[i][j],
                                       points_of_table[i][j + 1],
                                       points_of_table[i + 1][j],
                                       points_of_table[i + 1][j + 1], i, j) + ';')
                f.write('\n')

            print('End of page')
