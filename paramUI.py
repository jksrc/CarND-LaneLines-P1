import sys

import cv2
import numpy as np


class ParamUI:
    def __init__(self, target=None, p1=1, p2=1, p3=1, p1Title='p1', p2Title='p2', p3Title='p3'):
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        self._p1Title = p1Title
        self._p2Title = p2Title
        self._p3Title = p3Title
        self.atarget = target
        self.current_image = None
        self.title = 'test_out'

        def onchange2(pos):
            self._p2 = pos
            self._render()

        def onchange3(pos):
            self._p3 = pos
            self._render()

        def onchange1(pos):
            self._p1 = pos
            # self._p1 += (self._p1 + 1) % 2
            self._render()

        def capture_img(pos):
#            if self.current_image != None:
                print("write..")
                self.write_img()
                #cv2.setTrackbarPos('switch', 'Three param UI', 0)

        cv2.namedWindow('Three param UI')

        cv2.createTrackbar(self._p1Title, 'Three param UI', self._p1, 255, onchange1)
        cv2.createTrackbar(self._p2Title, 'Three param UI', self._p2, 255, onchange2)
        cv2.createTrackbar(self._p3Title, 'Three param UI', self._p3, 255, onchange3)

        switch = '0 : OFF \n1 : ON'
        self.buttonTrack = cv2.createTrackbar('save', 'Three param UI', 0, 1, capture_img)

        print "Adjust the parameters as desired.  Hit any key to close."

    def p2(self):
        return self._p2

    def p3(self):
        return self._p3

    def p1(self):
        return self._p1

    def image(self):
        return self.image

    def target(self):
        return self.target

    def setTarget(self, t):
        print("update target")
        self.atarget = t

    def setTitle(self, title):
        self.title = title


    def write_img(self):
        cv2.setTrackbarPos('save', 'Three param UI', 0)
        cv2.imwrite("./test_images_out/{}.jpg".format(self.title), self.current_image)



    def _render(self):
        image = self.atarget.update()
        self.current_image = image
        cv2.imshow('result', image)

    def start(self):
        self._render()

        cv2.waitKey(0)

        cv2.destroyWindow('Three param UI')
        cv2.destroyWindow('result')
        sys.exit()
