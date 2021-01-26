from PyQt5 import QtWidgets, QtGui, QtCore, Qt
import cv2
import math
import copy
import numpy as np
import os


class ImageScroller(QtWidgets.QWidget):
    def __init__(self, parent=None, pict_path=None):
        self.chosen_points = []
        self.corners = []
        self.draw_mode = 0
        QtWidgets.QWidget.__init__(self, parent)
        self.setGeometry(QtCore.QRect(0, 0, 630, 900))
        self.setFocusPolicy(Qt.Qt.StrongFocus)
        self.mouse_on = False
        self.render_coeff = 1.0
        self.pict_path = pict_path
        if pict_path is not None:
            self.pict = cv2.imdecode(np.fromfile(pict_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # cv2.imread(pict_path)
            self.path = QtGui.QPainterPath()

    def calc_size(self, parent_widget_size, img_size):
        coeffs = [parent_widget_size[0] / img_size[0], parent_widget_size[1] / img_size[1]]
        if coeffs[0] < coeffs[1]:
            self.render_coeff = coeffs[0]
        else:
            self.render_coeff = coeffs[1]

    def paintEvent(self, paint_event):
        self.calc_size([self.size().width(), self.size().height()], [self.pict.shape[1], self.pict.shape[0]])
        pict_render = cv2.resize(self.pict, (round(self.pict.shape[1] * self.render_coeff),
                                             round(self.pict.shape[0] * self.render_coeff)))
        self.resize(pict_render.shape[1], pict_render.shape[0])
        h, w, ch = pict_render.shape
        self._image = QtGui.QImage(pict_render.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        painter = QtGui.QPainter(self._image)
        pen = QtGui.QPen()
        pen.setWidth(self.width() // 200)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.draw_mode == 2 and len(self.corners) > 0:
            painter.drawLine(self.corners[0][0] * self.width(), self.corners[0][1] * self.height(),
                             self.corners[1][0] * self.width(), self.corners[0][1] * self.height())
            painter.drawLine(self.corners[1][0] * self.width(), self.corners[0][1] * self.height(),
                             self.corners[1][0] * self.width(), self.corners[1][1] * self.height())
            painter.drawLine(self.corners[1][0] * self.width(), self.corners[1][1] * self.height(),
                             self.corners[0][0] * self.width(), self.corners[1][1] * self.height())
            painter.drawLine(self.corners[0][0] * self.width(), self.corners[1][1] * self.height(),
                             self.corners[0][0] * self.width(), self.corners[0][1] * self.height())
        else:
            for pos in self.chosen_points:
                painter.drawLine(round(pos[0][0] * self.width()),
                                 round(pos[0][1] * self.height()),
                                 round(pos[1][0] * self.width()),
                                 round(pos[1][1] * self.height()))
        painter.drawPath(self.path)
        painter = QtGui.QPainter(self)
        painter.drawImage(paint_event.rect(), self._image, self.rect())

    def mousePressEvent(self, cursor_event):
        if self.draw_mode == 0:
            self.chosen_points.append([[cursor_event.pos().x() / (self.width()),
                                        cursor_event.pos().y() / (self.height())],
                                       [cursor_event.pos().x() / (self.width()),
                                        cursor_event.pos().y() / (self.height())]]
                                      )
        elif self.draw_mode == 2:
            self.corners = [[cursor_event.pos().x() / (self.width()), cursor_event.pos().y() / (self.height())],
                            [cursor_event.pos().x() / (self.width()), cursor_event.pos().y() / (self.height())]]
        self.mouse_on = True
        self.update()

    def mouseMoveEvent(self, cursor_event):
        if not self.mouse_on:
            return None
        if self.draw_mode == 0:
            self.chosen_points[-1][1] = [cursor_event.pos().x() / (self.width()),
                                         cursor_event.pos().y() / (self.height())]
        elif self.draw_mode == 2:
            self.corners[-1] = [cursor_event.pos().x() / (self.width()), cursor_event.pos().y() / (self.height())]
        self.update()

    def mouseReleaseEvent(self, cursor_event):
        if self.draw_mode == 0:
            self.mouse_on = False

            if len_seg([[self.chosen_points[-1][0][0] * self.width(),
                         self.chosen_points[-1][0][1] * self.height()],
                        [self.chosen_points[-1][1][0] * self.width(),
                         self.chosen_points[-1][1][1] * self.height()]]) < 10:
                del self.chosen_points[-1]
                return
            ind_first_vert_line = 0
            for i in range(len(self.chosen_points)):
                line = self.chosen_points[i]
                delta_x = math.fabs((line[0][0] - line[1][0]) * self.width())
                delta_y = math.fabs((line[0][1] - line[1][1]) * self.height())
                if delta_x > delta_y:
                    ind_first_vert_line = ind_first_vert_line + 1
                else:
                    break

            line = self.chosen_points[-1]
            delta_x = math.fabs((line[0][0] - line[1][0]) * self.width())
            delta_y = math.fabs((line[0][1] - line[1][1]) * self.height())
            if delta_x > delta_y:
                #  gorizontal line
                index = 0
                for i in range(0, ind_first_vert_line):
                    if self.chosen_points[i][0][1] < line[0][1]:
                        index = index + 1
                    else:
                        break
            else:
                index = ind_first_vert_line
                for i in range(ind_first_vert_line, len(self.chosen_points)):
                    if self.chosen_points[i][0][0] < line[0][0]:
                        index = index + 1
                    else:
                        break
            self.chosen_points.insert(index, copy.deepcopy(line))
            del self.chosen_points[-1]
        elif self.draw_mode == 1:
            if cursor_event.pos().x() > self.width() or cursor_event.pos().y() > self.height():
                return
            p = [cursor_event.pos().x(), cursor_event.pos().y()]
            ind = 0
            line = [[self.chosen_points[0][0][0] * self.width(), self.chosen_points[0][0][1] * self.height()],
                    [self.chosen_points[0][1][0] * self.width(), self.chosen_points[0][1][1] * self.height()]]
            min_dist = dist_between_point_and_seg(p, line)
            for i in range(1, len(self.chosen_points)):
                line = [[self.chosen_points[i][0][0] * self.width(), self.chosen_points[i][0][1] * self.height()],
                        [self.chosen_points[i][1][0] * self.width(), self.chosen_points[i][1][1] * self.height()]]
                if dist_between_point_and_seg(p, line) < min_dist:
                    ind = i
                    min_dist = dist_between_point_and_seg(p, line)
            del self.chosen_points[ind]
        self.update()

    def getPixmapImage(self):
        return self._image

    def saveImage(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '', '*.jpg')[0]
        # painter = QtGui.QPainter()
        # painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # self.render(painter)
        self._image.save(fileName)

    def setImage(self, pict_path):
        self.pict = cv2.imdecode(np.fromfile(pict_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # cv2.imread(pict_path)
        self.chosen_points = []
        self.corners = []
        self.pict_path = pict_path
        self.path = QtGui.QPainterPath()
        self.update()

    def setLines(self, lines):
        self.chosen_points = lines
        self.update()

    def getLines(self, lines):
        return self.chosen_points

    def keyPressEvent(self, event):
        if event.key() == Qt.Qt.Key_Space:
            self.saveImage()
        if event.key() == Qt.Qt.Key_Backspace and len(self.chosen_points) > 0:
            del self.chosen_points[-1]
            self.update()
        if event.key() == Qt.Qt.Key_C and len(self.corners) > 0:
            self.pict = self.pict[int(self.corners[0][1] * self.pict.shape[0]):int(self.corners[1][1] * self.pict.shape[0]),
                                  int(self.corners[0][0] * self.pict.shape[1]):int(self.corners[1][0] * self.pict.shape[1])]
            self.corners = []
            self.update()
            extention = "." + os.path.splitext(self.pict_path)[1][1:]
            success, im_buf_arr = cv2.imencode(extention, self.pict)
            if success:
                im_buf_arr.tofile(self.pict_path)
            self.corners = []
            self.update()


def len_seg(seg):
    return math.sqrt(pow(seg[1][0] - seg[0][0], 2) + pow(seg[1][1] - seg[0][1], 2))


def dist_between_point_and_seg(p, seg):
    a = len_seg([seg[0], p])
    b = len_seg([p, seg[1]])
    c = len_seg(seg)

    ax = p[0] - seg[0][0]
    ay = p[1] - seg[0][1]
    bx = seg[1][0] - p[0]
    by = seg[1][1] - p[1]

    return math.sqrt(pow(a * b, 2) - pow(ax * bx + ay * by, 2)) / c


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = ImageScroller()
    # w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())
