# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2013-2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment

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

from ..core.profile import Profile
from ..core.temporalscalarfields import TemporalScalarFields
from ..core.scalarfield import ScalarField
from ..utils.progresscounter import ProgressCounter
from ..plotlib import get_color_cycles
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Stabilizer(object):
    """
    Stabilizer for TemporalFields, based on ORB implementation of opencv.
    """

    def __init__(self, obj, orb_kwargs={}, mode='continuous'):
        """
        Stabilizer for TemporalFields, based on ORB implementation of opencv.

        Parameters
        ==========
        obj: TemporalFields object
            Fields on wich performing the stabilization.
        orb_kwargs: dic
            Additional arguments for the ORB feature detectior
            (See opencv documentation for more information)
        mode: string in ['continuous', 'from_first']
            If 'continuous', stabilize each frame in regards of the previous
            one.
            If 'from_first', stabilize all frame from the first one.
        """
        self.obj = obj
        self.kp_kwargs = orb_kwargs
        self.kp_detector = cv2.ORB_create(**orb_kwargs)
        self.kp_keypoints = []
        self.mode = mode
        self.raw_transform = None
        self.smoothed_transform = None
        self.raw_trajectory = None
        self.smoothed_trajectory = None
        self.stabilized_obj = None

    def _compute_transform(self, dense=False, opt_flow_args={}, verbose=False):
        """
        Use the feature points to compute the deformation transformation for
        each frames.

        Parameters
        ==========
        dense: boolean
            If True, use dens optical flow (track every pixel instead
            of tracking features). This method give better results when the
            frames don't show well-defined features, but is slower.
        opt_flow_args: dic
            Additional arguments for the optical flow functions
            ('calcOpticalFLowPyrLK' or 'calcOpticalFlowFarneback',
            see opencv documentation.
        """
        # verbose
        if verbose:
            pg = ProgressCounter(init_mess="Computing transforms",
                                 nmb_max=len(self.obj) - 1,
                                 name_things="transforms")
        prev_to_cur_transform = []
        prev_im = self.obj[0].values
        for i in np.arange(1, len(self.obj)):
            cur_im = self.obj[i].values
            if dense:
                raise Exception('Not working for drops...')
                args = {'pyr_scale': 0.5,
                        'levels': 3,
                        'winsize': 15,
                        'iterations': 3,
                        'poly_n': 5,
                        'poly_sigma': 1.2,
                        'flags': 0}
                args.update(opt_flow_args)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_im,
                    cur_im,
                    None, **args)
                # get mean displacement from dxs, dys
                dxs = flow[..., 0]
                dys = flow[..., 1]
                dx = np.median(dxs)
                dy = np.median(dys)
                da = 0
            else:
                # detect keypoints
                prev_kps = self.kp_detector.detect(prev_im)
                self.kp_keypoints.append(prev_kps)
                if len(prev_kps) == 0:
                    raise Exception(f'No features found on the {i}th frame, '
                                    'you should try reducing the '
                                    '\'fastThreshold\' value.')
                prev_kps = np.array(
                    [kp.pt for kp in prev_kps],
                    dtype='float32').reshape(-1, 1, 2)
                # calc flow of movement
                cur_kps, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_im,
                    cur_im,
                    prev_kps,
                    None,
                    **opt_flow_args)
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
            if self.mode == 'continuous':
                # set current frame to prev frame for use in next iteration
                prev_im = cur_im
            # verbose
            if verbose:
                pg.print_progress()
        # store resulting traj and transform
        if self.mode == 'continuous':
            self.raw_transform = np.array(prev_to_cur_transform)
            self.raw_trajectory = np.cumsum(prev_to_cur_transform, axis=0)
        elif self.mode == 'from_first':
            prev_to_cur_transform = np.array(prev_to_cur_transform)
            self.raw_transform = np.concatenate(([prev_to_cur_transform[0]],
                                                 prev_to_cur_transform[1::] -
                                                 prev_to_cur_transform[0:-1]))
            self.raw_trajectory = np.array(prev_to_cur_transform)
        else:
            raise ValueError()

    def _smooth_transform(self, smooth_size=30, verbose=False):
        """
        Smooth the computed transformations to get smoother transitions in the
        resulting video.

        Parameters
        ==========
        smooth_size: integer
            Size of the smoothing to use (default to 30)
        """
        # check
        if self.raw_transform is None:
            raise Exception('You should compute the transformation first.')
        # verbose
        if verbose:
            print("=== Smoothing transforms ===")
        self.smoothed_trajectory = self.raw_trajectory.copy()
        for i in range(3):
            tmp_prof = Profile(self.obj.times[:-1], self.raw_trajectory[:, i])
            tmp_prof.smooth(tos='gaussian', size=smooth_size,
                            inplace=True)
            self.smoothed_trajectory[:, i] = tmp_prof.y
        self.smoothed_transform = (self.raw_transform
                                   + (self.smoothed_trajectory
                                      - self.raw_trajectory))

    def _apply_transform(self, border_type='black', border_size=0,
                         verbose=False):
        """
        Apply the computed transformation to the fields.

        Parameters:
        ===========
        border_type: string in ['black', 'reflect', 'replicate']
            Type of borders.
        border_size: integer
            Size in pixels of the additional border to use
            (to avoid loosing part of the frames during the transformation).
        """
        # checks
        if self.smoothed_transform is None:
            raise Exception('You should compute and smooth the transformation'
                            ' first.')
        # verbose
        if verbose:
            pg = ProgressCounter(init_mess="Applying transform",
                                 nmb_max=len(self.obj),
                                 name_things="transforms")
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
        transf = np.concatenate(([[0, 0, 0]], self.smoothed_transform))
        for i in range(len(self.obj)):
            # build transformation matrix
            transform = np.zeros((2, 3))
            transform[0, 0] = np.cos(transf[i][2])
            transform[0, 1] = -np.sin(transf[i][2])
            transform[1, 0] = np.sin(transf[i][2])
            transform[1, 1] = np.cos(transf[i][2])
            transform[0, 2] = transf[i][0]
            transform[1, 2] = transf[i][1]
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
            transformed_im = transformed_im[border_size:(transformed_im.shape[0]
                                                         - border_size),
                                            border_size:(transformed_im.shape[1]
                                                         - border_size)]
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
                              unit_times=self.obj.unit_times,
                              copy=False)
            # verbose
            if verbose:
                pg.print_progress()
        self.stabilized_obj = res_tsf

    def _apply_transform_to_point(self, pts, from_frame=0, verbose=True):
        """
        Apply the transformation to pts of the first image.

        Parameters
        ==========
        pts: Nx2 array
            Point of the first array
        from_frame: integer
            Frame from wich the tracking should take its source.
        """
        # checks
        if self.smoothed_transform is None:
            raise Exception('You should compute and smooth the transformation'
                            ' first.')
        transform = self.raw_transform
        pts = np.array(pts)
        if pts.ndim == 1:
            pts = np.array([pts])
        if from_frame != 0:
            raise Exception('Not implemented yet')
        # verbose
        if verbose:
            pg = ProgressCounter(init_mess="Tracking points motion",
                                 nmb_max=len(self.obj) - 1,
                                 name_things="transforms")
            n = 0
        all_new_pts = []
        for pt in pts:
            new_pts = [np.array(pt)]
            for tr in transform:
                tr_mat = np.zeros((3, 3))
                tr_mat[0, 0] = np.cos(tr[2])
                tr_mat[0, 1] = -np.sin(tr[2])
                tr_mat[0, 2] = 0
                tr_mat[1, 0] = np.sin(tr[2])
                tr_mat[1, 1] = np.cos(tr[2])
                tr_mat[1, 2] = 0
                tr_mat[0, 2] = tr[0]
                tr_mat[1, 2] = tr[1]
                tr_mat[2, 2] = 1
                pt = np.array([new_pts[-1][1], new_pts[-1][0]],
                              dtype=np.float32)
                pt = np.reshape(pt, (1, 1, 2))
                tmp_pts = cv2.perspectiveTransform(pt, tr_mat)
                tmp_pts = tmp_pts[0][0][0:2]
                tmp_pts = tmp_pts[::-1]
                new_pts.append(tmp_pts)
                # verbose
                if verbose:
                    n += 1
                    if n%len(pts) == 0:
                        pg.print_progress()
            all_new_pts.append(new_pts)
        # return
        return np.array(all_new_pts)

    def get_stabilized_obj(self, dense=False, smooth_size=30,
                           border_type='black', border_size=0,
                           verbose=False):
        """
        Stabilize the fields.

        Parameters
        ==========
        dense: boolean
            If True, use dens optical flow (track every pixel instead
            of tracking features).
        smooth_size: integer
            Size of the smoothing to use (default to 30)
        border_type: string in ['black', 'reflect', 'replicate']
            Type of borders.
        border_size: integer
            Size in pixels of the additional border to use
            (to avoid loosing part of the frames during the transformation).
        """
        self._compute_transform(dense=dense, verbose=verbose)
        self._smooth_transform(smooth_size=smooth_size, verbose=verbose)
        self._apply_transform(border_type=border_type, border_size=border_size,
                              verbose=verbose)
        return self.stabilized_obj

    def display_transform(self):
        """
        """
        # colors
        colors = get_color_cycles()
        # data
        data_dic = {'Raw transform': self.raw_transform,
                    'Raw trajectory': self.raw_trajectory,
                    'Smoothed trajectory': self.smoothed_trajectory,
                    'Smoothed transform': self.smoothed_transform}
        # plot
        for name, data in data_dic.items():
            fig, ax = plt.subplots(2, 1, sharex=True)
            plt.sca(ax[0])
            plt.plot(data[:, 0], label=f"{name} x",
                     color=colors[0])
            plt.plot(data[:, 1], label=f"{name} y",
                     color=colors[1])
            plt.legend()
            plt.sca(ax[1])
            plt.plot(data[:, 2], label=f"{name} $\\theta$",
                     color=colors[2])
            plt.legend()
