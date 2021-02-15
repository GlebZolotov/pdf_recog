from os.path import exists
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path
import os
import shutil
import configparser
import logging
import cv2 as cv
import numpy as np
import pdf_recog
import copy
import math


def check_copy(name_img):
    if not os.path.exists(name_img[:-4] + "_s.jpg"):
        shutil.copyfile(name_img, name_img[:-4] + "_s.jpg")
    return name_img[:-4] + "_s.jpg"


def save_cv_img(img, name):
    ext = "." + os.path.splitext(name)[1][1:]
    success, im_buf_arr = cv.imencode(ext, img)
    if success:
        im_buf_arr.tofile(name)


def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
    return outImg


def open_cv_img(name):
    return cv.imdecode(np.fromfile(name, dtype=np.uint8), cv.IMREAD_GRAYSCALE)


class Model:
    def __init__(self):
        self.fileName = None
        self.foldName = None
        self.count_of_pages = 0
        self.logger = logging.getLogger("startApp")
        self.lines = []
        self.shape = []

    def isValid(self, file_name):
        return exists(file_name)

    def setFileName(self, fileName):
        if self.isValid(fileName):
            self.fileName = fileName
            self.foldName = os.sep.join(os.path.normpath(fileName).split(os.sep)[:-2])

    def setFileNamePDF(self, fileName):
        self.setFileName(fileName)
        if self.fileName == fileName:
            self.foldName = os.path.splitext(self.fileName)[0]

    def getFileName(self):
        return self.fileName

    def load_file(self, name_fold=""):
        cf = configparser.ConfigParser()
        cf.read("settings.ini")
        if name_fold == "" and not (self.foldName is None):
            name_fold = self.foldName
        if os.path.exists(name_fold):
            shutil.rmtree(name_fold)
        os.mkdir(name_fold)
        name_source_fold = os.path.join(name_fold, "Source")
        os.mkdir(name_source_fold)
        name_table_fold = os.path.join(name_fold, "Tables")
        os.mkdir(name_table_fold)
        self.logger.info("start convert")
        with TemporaryDirectory() as tmpdir:
            i_ = 1
            names_img = []
            # with open(self.fileName.replace(".pdf", ".csv"), 'w') as f: #
            for img in convert_from_path(self.fileName, output_folder=tmpdir,
                                         poppler_path=cf.get("paths", "poppler_path")):
                names_img.append(os.path.join(name_source_fold, str(i_) + ".jpg"))
                img.save(names_img[-1], 'JPEG')
                # f.write(str(i_) + ";" + pdf_recog.recog_text_full(open_cv_img(names_img[-1])) + '\n')  # jhg
                i_ = i_ + 1
        self.logger.info("end convert")
        return names_img

    def get_lines_of_table(self, name_img, params):
        cv_img = cv.imdecode(np.fromfile(name_img, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
        self.shape = [cv_img.shape[1], cv_img.shape[0]]
        # table = pdf_recog.cut_table(cv_img)
        points_of_table = pdf_recog.detect_lines(cv_img, params)
        corners = [points_of_table[0][0], points_of_table[0][-1],
                   points_of_table[-1][0], points_of_table[-1][-1]]

        corns = np.copy(cv_img)
        cv.line(corns, (corners[0][1], corners[0][0]), (corners[1][1], corners[1][0]), (0, 0, 0), 10)
        cv.line(corns, (corners[0][1], corners[0][0]), (corners[2][1], corners[2][0]), (0, 0, 0), 10)
        cv.line(corns, (corners[3][1], corners[3][0]), (corners[1][1], corners[1][0]), (0, 0, 0), 10)
        cv.line(corns, (corners[3][1], corners[3][0]), (corners[2][1], corners[2][0]), (0, 0, 0), 10)
        # cv.imwrite(name_img.replace("Source", "Tables"), corns)

        thresh_ = cv.threshold(cv_img, params.thr_border, 255, cv.THRESH_BINARY_INV)[1]
        # Поиск линий таблицы
        upper_lines = pdf_recog.clean_points(pdf_recog.fill_lines(thresh_, corners[0], corners[1], 0, params), params.min_dist_between_points)
        left_lines = pdf_recog.clean_points(pdf_recog.fill_lines(thresh_, corners[0], corners[2], 1, params), params.min_dist_between_points)
        right_lines = pdf_recog.clean_points(pdf_recog.fill_lines(thresh_, corners[1], corners[3], 2, params,
                                                                  [corners[1][0] - corners[0][0], left_lines]), params.min_dist_between_points)
        down_lines = pdf_recog.clean_points(pdf_recog.fill_lines(thresh_, corners[2], corners[3], 3, params,
                                                                 [corners[2][1] - corners[0][1], upper_lines]), params.min_dist_between_points)

        if len(upper_lines) == len(down_lines):
            self.logger.info('OK up down')
        else:
            self.logger.info('Error up down ' + str(len(upper_lines)) + ' ' + str(len(down_lines)))

        if len(left_lines) == len(right_lines):
            self.logger.info('OK left right')
        else:
            self.logger.info('Error left right ' + str(len(left_lines)) + ' ' + str(len(right_lines)))
            self.logger.info(left_lines)
            self.logger.info(right_lines)
        gor_lines = [[copy.deepcopy(corners[0]), copy.deepcopy(corners[1])]] + [[left_lines[i], right_lines[i]] for i in range(len(left_lines))] + [
            [copy.deepcopy(corners[2]), copy.deepcopy(corners[3])]]
        vert_lines = [[copy.deepcopy(corners[0]), copy.deepcopy(corners[2])]] + [[upper_lines[i], down_lines[i]] for i in range(len(upper_lines))] + [
            [copy.deepcopy(corners[1]), copy.deepcopy(corners[3])]]

        res_lines = gor_lines + vert_lines

        corns = np.copy(cv_img)
        for line in res_lines:
            cv.line(corns, (line[0][1], line[0][0]), (line[1][1], line[1][0]),
                    (0, 0, 0),
                    10)
        extention = "." + os.path.splitext(name_img)[1][1:]
        success, im_buf_arr = cv.imencode(extention, corns)
        if success:
            im_buf_arr.tofile(name_img.replace("Source", "Tables"))
        return [gor_lines, vert_lines]

    def recog_text(self, gor_lines, vert_lines, name_img, params):
        cv_img = cv.imdecode(np.fromfile(name_img, dtype=np.uint8), cv.IMREAD_GRAYSCALE)

        # Расчёт матрицы координат ячеек таблицы
        points_of_table = [[line[0] for line in vert_lines]]
        for i in range(1, len(gor_lines) - 1):
            points_of_table.append([gor_lines[i][0]] + [[0, 0]] * (len(vert_lines) - 2) + [gor_lines[i][1]])
        points_of_table.append([line[1] for line in vert_lines])

        down_limit = len(points_of_table) - 1
        right_limit = len(points_of_table[0]) - 1
        for i in range(1, down_limit):
            for j in range(1, right_limit):
                points_of_table[i][j] = pdf_recog.point_of_intersect(
                    points_of_table[0][j],
                    points_of_table[down_limit][j],
                    points_of_table[i][0],
                    points_of_table[i][right_limit])

        page_counter = os.path.basename(name_img).split(".")[0]
        with open(name_img.replace(".jpg", ".csv"), 'w') as f:
            for i in range(down_limit):
                f.write(str(page_counter) + ";")
                for j in range(right_limit):
                    f.write(pdf_recog.recog_text(cv_img, points_of_table[i][j],
                                                 points_of_table[i][j + 1],
                                                 points_of_table[i + 1][j],
                                                 points_of_table[i + 1][j + 1], i, j, params) + ';')
                f.write('\n')

    def recog_imgs(self, names_img, params):
        for name_img in names_img:
            self.detect_table(name_img, params)
            if self.lines is None:
                continue
            self.recog_table(name_img, params)
            self.logger.info("End " + name_img)

    def detect_table(self, name_img, params):
        self.logger.info("Start " + name_img)
        try:
            if not params.emulate_border:
                self.lines = self.get_lines_of_table(name_img, params)
            else:
                self.lines = self.emulate_lines_of_table(name_img, params)
            self.logger.info("Detect table " + name_img)
        except Exception:
            self.logger.error("Error in detect table")
            self.logger.error("Exception occurred", exc_info=True)
            self.lines = None

    def recog_table(self, name_img, params):
        self.logger.info("Start recog" + name_img)
        try:
            self.recog_text(self.lines[0], self.lines[1], name_img, params)
            self.logger.info("Recognition table complete " + name_img)
        except Exception:
            self.logger.error("Error in recog table")
            self.logger.error("Exception occurred", exc_info=True)

    def lines_for_drawing(self):
        if self.lines is None:
            return []
        res_lines = self.lines[0] + self.lines[1]
        for i in range(len(res_lines)):
            x1 = res_lines[i][0][1]
            y1 = res_lines[i][0][0]
            x2 = res_lines[i][1][1]
            y2 = res_lines[i][1][0]
            res_lines[i][0][0] = x1 / self.shape[0]
            res_lines[i][0][1] = y1 / self.shape[1]
            res_lines[i][1][0] = x2 / self.shape[0]
            res_lines[i][1][1] = y2 / self.shape[1]
        return res_lines

    def lines_for_recog(self, lines):
        gor_lines = []
        vert_lines = []
        if len(self.shape) == 0:
            cv_img = cv.imdecode(np.fromfile(self.fileName, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
            self.shape = [cv_img.shape[1], cv_img.shape[0]]
        for i in range(len(lines)):
            line = lines[i]
            x1 = line[0][1]
            y1 = line[0][0]
            x2 = line[1][1]
            y2 = line[1][0]
            delta_x = math.fabs((line[0][0] - line[1][0]) * self.shape[0])
            delta_y = math.fabs((line[0][1] - line[1][1]) * self.shape[1])
            if delta_x > delta_y:
                gor_lines.append([[round(x1 * self.shape[1]), round(y1 * self.shape[0])],
                                  [round(x2 * self.shape[1]), round(y2 * self.shape[0])]])
            else:
                vert_lines.append([[round(x1 * self.shape[1]), round(y1 * self.shape[0])],
                                   [round(x2 * self.shape[1]), round(y2 * self.shape[0])]])
        self.lines = [gor_lines, vert_lines]

    def handle_img(self, name_img, params):
        cv_img = open_cv_img(check_copy(name_img))
        cv_img = rotation(cv_img, params.angle / 5)
        if params.cut_width > 0:
            cv_img = cv_img[params.cut_width:-params.cut_width, params.cut_width:-params.cut_width]
        save_cv_img(cv_img, name_img)

    # type_line: 0 - gorizontal, 1 - vertical
    def check_empty_line(self, cv_thr, coord, start_c, stop_c, type_line):
        if type_line == 0:
            for y in range(start_c, stop_c):
                if cv_thr[coord, y] == 0:
                    return False
        elif type_line == 1:
            for x in range(start_c, stop_c):
                if cv_thr[x, coord] == 0:
                    return False
        return True

    def emulate_lines_of_table(self, name_img, params):
        cv_img = open_cv_img(name_img)
        self.shape = [cv_img.shape[1], cv_img.shape[0]]
        cv_thr = cv.threshold(cv_img, params.thr_border, 255, cv.THRESH_BINARY)[1]
        cv.imwrite("imgs\\thr0.jpg", cv_thr)

        h, w = cv_thr.shape[:2]
        i_start = 5
        while self.check_empty_line(cv_thr, i_start, 0, w, 0):
            i_start = i_start + 5
        i_start = i_start - 5
        i_stop = h - 1 - 5
        while self.check_empty_line(cv_thr, i_stop, 0, w, 0):
            i_stop = i_stop - 5
        i_stop = i_stop + 5
        j_start = 5
        while self.check_empty_line(cv_thr, j_start, 0, h, 1):
            j_start = j_start + 5
        j_start = j_start - 5
        j_stop = w - 1 - 5
        while self.check_empty_line(cv_thr, j_stop, 0, h, 1):
            j_stop = j_stop - 5
        j_stop = j_stop + 5

        gor_lines = [[[i_start, j_start], [i_start, j_stop]]]
        vert_lines = [[[i_start, j_start], [i_stop, j_start]]]

        i = i_start + 5
        while i < i_stop - 5:
            while i < i_stop - 5 and not self.check_empty_line(cv_thr, i, j_start, j_stop, 0):
                i = i + 5
            if i >= i_stop - 5:
                break
            i1 = i
            while i < i_stop - 5 and self.check_empty_line(cv_thr, i, j_start, j_stop, 0):
                i = i + 1
            if i >= i_stop - 5:
                break
            gor_lines.append([[(i1 + i - 1) // 2, j_start], [(i1 + i - 1) // 2, j_stop]])

        j = j_start + 5
        while j < j_stop - 5:
            while j < j_stop - 5 and not self.check_empty_line(cv_thr, j, i_start, i_stop, 1):
                j = j + 5
            if j >= j_stop - 5:
                break
            j1 = j
            while j < j_stop - 5 and self.check_empty_line(cv_thr, j, i_start, i_stop, 1):
                j = j + 1
            if j >= j_stop - 5:
                break
            vert_lines.append([[i_start, (j1 + j - 1) // 2], [i_stop, (j1 + j - 1) // 2]])

        gor_lines.append([[i_stop, j_start], [i_stop, j_stop]])
        vert_lines.append([[i_start, j_stop], [i_stop, j_stop]])
        return [gor_lines, vert_lines]



