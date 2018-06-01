# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2013-2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment
# Version: 1.0

# This file is part of IMTreatment.

# IMTreatment is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# IMTreatment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This submodule is inspired from https://github.com/galaunay/python_video_stab

import imutils.feature.factories as kp_factory
from ..core.profile import Profile
from ..core.temporalscalarfields import TemporalScalarFields
from ..core.scalarfield import ScalarField
import numpy as np
import cv2


class Stabilizer(object):

    def __init__(self, obj, kp_method='ORB', kp_kwargs={}):
        """
        """
        self.obj = obj
        self.kp_method = kp_method
        self.kp_kwargs = kp_kwargs
        self.kp_detector = kp_factory.FeatureDetector_create(
            kp_method, **kp_kwargs)
        self.raw_transform = None
        self.raw_trajectory = None
        self.smoothed_trajectory = None
        self.stabilized_obj = None

    def compute_transform(self):
        """
        """
        prev_to_cur_transform = []
        prev_im = self.obj[0].values
        for i in range(len(self.obj)):
            cur_im = self.obj[i].values
            # detect keypoints
            prev_kps = self.kp_detector.detect(prev_im)
            prev_kps = np.array(
                [kp.pt for kp in prev_kps],
                dtype='float32').reshape(-1, 1, 2)
            # calc flow of movement
            cur_kps, status, err = cv2.calcOpticalFlowPyrLK(
                prev_im,
                cur_im,
                prev_kps,
                None)
            # storage for keypoints with status 1
            prev_matched_kp = []
            cur_matched_kp = []
            for i, matched in enumerate(status):
                # store coords of keypoints that appear in both
                if matched:
                    prev_matched_kp.append(prev_kps[i])
                    cur_matched_kp.append(cur_kps[i])
            # estimate partial transform
            transform = cv2.estimateRigidTransform(
                np.array(prev_matched_kp),
                np.array(cur_matched_kp),
                False)
            if transform is not None:
                dx = transform[0, 2]
                dy = transform[1, 2]
                da = np.arctan2(transform[1, 0], transform[0, 0])
            else:
                dx = dy = da = 0

            # store transform
            prev_to_cur_transform.append([dx, dy, da])
            # set current frame to prev frame for use in next iteration
            prev_im = cur_im
        # convert list of transforms to array
        self.raw_transform = np.array(prev_to_cur_transform)
        # cumsum of all transforms for trajectory
        self.raw_trajectory = np.cumsum(prev_to_cur_transform, axis=0)

    def smooth_transform(self, smooth_size=30):
        """
        """
        self.smoothed_trajectory = self.raw_trajectory.copy()
        for i in range(3):
            tmp_prof = Profile(self.obj.times, self.raw_trajectory[:, i])
            tmp_prof.smooth(tos='gaussian', size=smooth_size,
                            inplace=True)
            self.smoothed_trajectory[:, i] = tmp_prof.y
        self.smoothed_transform = (self.raw_transform +
                                   (self.smoothed_trajectory -
                                    self.raw_trajectory))

    def apply_transform(self, border_type='black', border_size=0):
        """
        """
        # checks
        border_modes = {
            'black': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE
        }
        border_mode = border_modes[border_type]
        # get im shape
        h, w = self.obj.shape
        h += 2 * border_size
        w += 2 * border_size
        # create result holder
        res_tsf = TemporalScalarFields()
        dx = self.obj.dx
        x0, xf = self.obj.axe_x[0], self.obj.axe_x[-1]
        dy = self.obj.dy
        y0, yf = self.obj.axe_y[0], self.obj.axe_y[-1]
        axe_x = np.arange(x0 - border_size*dx, xf + border_size*dx + dx, dx)
        axe_y = np.arange(y0 - border_size*dy, yf + border_size*dy + dx, dy)
        # main loop
        for i in range(len(self.obj)):
            # build transformation matrix
            transform = np.zeros((2, 3))
            transform[0, 0] = np.cos(self.smoothed_transform[i][2])
            transform[0, 1] = -np.sin(self.smoothed_transform[i][2])
            transform[1, 0] = np.sin(self.smoothed_transform[i][2])
            transform[1, 1] = np.cos(self.smoothed_transform[i][2])
            transform[0, 2] = self.smoothed_transform[i][0]
            transform[1, 2] = self.smoothed_transform[i][1]
            # apply transform
            bordered_im = cv2.copyMakeBorder(
                self.obj[i].values,
                top=border_size * 2,
                bottom=border_size * 2,
                left=border_size * 2,
                right=border_size * 2,
                borderType=border_mode,
                value=[0, 0, 0])
            transformed_im = cv2.warpAffine(
                bordered_im,
                transform,
                (w + border_size * 2, h + border_size * 2),
                borderMode=border_mode)
            transformed_im = transformed_im[border_size:(transformed_im.shape[0] - border_size),
                                            border_size:(transformed_im.shape[1] - border_size)]
            # store
            im = ScalarField()
            im.import_from_arrays(axe_x=axe_x,
                                  axe_y=axe_y,
                                  values=transformed_im,
                                  unit_x=self.obj.unit_x,
                                  unit_y=self.obj.unit_y,
                                  unit_values=self.obj.unit_values,
                                  dtype=np.uint8)
            res_tsf.add_field(im, time=self.obj.times[i],
                              unit_times=self.obj.unit_times)
        self.stabilized_obj = res_tsf

    def get_stabilized_obj(self, smooth_size=30, border_type='black',
                           border_size=0):
        """
        """
        self.compute_transform()
        self.smooth_transform(smooth_size=smooth_size)
        self.apply_transform(border_type=border_type, border_size=border_size)
        return self.stabilized_obj
