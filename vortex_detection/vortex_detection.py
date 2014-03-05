# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:07:07 2014

@author: muahah
"""

import pdb
import IMTreatment as imt
from ..core import Points, ScalarField, VectorField, make_unit,\
    ARRAYTYPES, NUMBERTYPES, VelocityField, TemporalVelocityFields
import matplotlib.pyplot as plt
import numpy as np
import sets


def find_critical_points_traj(TVFS, windows_size=5, radius=None, epsilon=None):
    """
    For a set of velocity field (TemporalVelocityFields object), return the
    trajectory of critical points.

    Parameters
    ----------
    TVFS : TemporalVelocityFields object
    windows_size : int, optional
        Size of the windows for isolating structures.
        (a small value gave less VF_others, but can lead to errors because of
        critical points no-finding).
    epsilon : float, optional
        Maximum length between two consecutive points in trajectory.
        (default is Inf), extremely usefull to put a correct value here.

    Returns
    -------
    rot_vc_traj : tuple of Points objects
        Rotative vortex centers trajectories
    contrarot_vc : tuple of Points objects
        Contrarotative vortex centers trajectories
    saddle_pts : tuple of Points objects
        Saddle points trajectories
    VF_others : tuple of VelocityField
        Fields zones contening multiples critical points
        (need advances treatment)
    """
    if not isinstance(TVFS, TemporalVelocityFields):
        raise TypeError("'TVFS' must be a TemporalVelocityFields")
    focus = []
    focus_c = []
    nodes_i = []
    nodes_o = []
    saddles = []
    others = []
    # getting critical points for all fields
    for field in TVFS.fields:
        foc, foc_c, nod_i, nod_o, sadd, oth\
            = find_critical_points(field, windows_size, radius)
        if len(foc) != 0:
            focus += foc
        if len(foc_c) != 0:
            focus_c += foc_c
        if len(nod_i) != 0:
            nodes_i += nod_i
        if len(nod_o) != 0:
            nodes_o += nod_o
        if len(sadd) != 0:
            saddles += sadd
        if len(oth) != 0:
            others += oth
    # getting critical points trajectory
    if len(focus) != 0:
        focus_traj = vortices_evolution(focus, epsilon)
    else:
        focus_traj = []
    if len(focus_c) != 0:
        focus_c_traj = vortices_evolution(focus_c, epsilon)
    else:
        focus_c_traj = []
    if len(nodes_i) != 0:
        nodes_i_traj = vortices_evolution(nodes_i, epsilon)
    else:
        nodes_i_traj = []
    if len(nodes_o) != 0:
        nodes_o_traj = vortices_evolution(nodes_o, epsilon)
    else:
        nodes_o_traj = []
    if len(saddles) != 0:
        saddles_traj = vortices_evolution(saddles, epsilon)
    else:
        saddles_traj = []
    return focus_traj, focus_c_traj, nodes_i_traj, nodes_o_traj, saddles_traj,\
        others


def find_critical_points(VF, windows_size=5, radius=None):
    """
    For a velocity field (VelocityField object), return the position of
    critical points.

    Parameters
    ----------
    VF : VelocityField object
    windows_size : int
        size (in field point) of the PBI windows. Small value tend to more
        critical points discovered, but with less accuracy.
        (To small values may lead to errors in criterion computation)
    radius : number
        radius for criterion computation (default lead to 8 points zone of
        computation)
        

    Returns
    -------
    focus, focus_c, nodes_i, nodes_o, saddles : tuple of points
        Found points of each type.
    other_VF : tuple of VelocityField
        Areas where PBI fail to isolate only one critical points
        (diminue with small 'windows_size')
    """
    if not isinstance(VF, VelocityField):
        raise TypeError("'VF' must be a VelocityField")
    if windows_size < 4:
        raise ValueError("'windows_size' is too small (minimum 4)")
    # isolating the interesting zones (contening critical points)
    VF_tupl = vortices_zoi(VF, windows_size=windows_size)
    # sorting by critical points type
    VF_focus = []
    VF_nodes = []
    VF_saddle = []
    VF_others = []
    for VF in VF_tupl:
        # test zoi size
        axe_x, axe_y = VF.get_axes()
        if len(axe_x) < 5 or len(axe_y) < 5:
            continue
        # test if node or focus
        if VF.PBI == 1:
            # checking if node or focus
            VF.gamma1 = get_gamma1(VF.V, radius)
            VF.kappa1 = get_kappa1(VF.V, radius)
            max_gam = np.max([abs(VF.gamma1.get_max()),
                             abs(VF.gamma1.get_min())])
            max_kap = np.max([abs(VF.kappa1.get_max()),
                             abs(VF.kappa1.get_min())])
            if max_gam > max_kap:
                VF_focus.append(VF)
            else:
                VF_nodes.append(VF)
        # saddle point
        elif VF.PBI == -1:
            VF_saddle.append(VF)
        # complicated
        elif VF.PBI == 0:
            VF_others.append(VF)
        else:
            raise StandardError("Uncorrect PBI value : {}".format(VF.PBI))
    # computing saddle points positions
    saddles = []
    if len(VF_saddle) != 0:
        for VF in VF_saddle:
            # getting iota
            V_tmp = VF.V
            tmp_iota = get_iota(V_tmp)
            pts = tmp_iota.get_zones_centers(bornes=[.75, 1])
            if pts is not None:
                if pts.xy[0][0] is not None and len(pts) == 1:
                    pts.v = [VF.time]
                    pts.unit_v = VF.unit_time
                    saddles.append(pts)
    # computing focus positions (rotatives and contrarotatives)
    focus = []
    focus_c = []
    if len(VF_focus) != 0:
        for VF in VF_focus:
            tmp_gam = VF.gamma1
            tmp_gam.smooth()
            min_gam = tmp_gam.get_min()
            max_gam = tmp_gam.get_max()
            # rotative vortex
            if abs(max_gam) > abs(min_gam):
                pts = tmp_gam.get_zones_centers(bornes=[0.4, 1], rel=False)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        focus.append(pts)
            # contrarotative vortex
            else:
                pts = tmp_gam.get_zones_centers(bornes=[-1, -0.4], rel=False)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        focus_c.append(pts)
    # computing nodes points positions (in or out)
    nodes_i = []
    nodes_o = []
    if len(VF_nodes) != 0:
        for VF in VF_nodes:
            tmp_kap = VF.kappa1
            min_kap = tmp_kap.get_min()
            max_kap = tmp_kap.get_max()
            # out nodes
            if abs(max_kap) > abs(min_kap):
                pts = tmp_kap.get_zones_centers(bornes=[0.75, 1], rel=False)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        nodes_o.append(pts)
            else:
                # in nodes
                pts = tmp_kap.get_zones_centers(bornes=[-1, -0.75], rel=False)
                if pts is not None:
                    if pts.xy[0][0] is not None and len(pts) == 1:
                        pts.v = [VF.time]
                        pts.unit_v = VF.unit_time
                        nodes_i.append(pts)
    return focus, focus_c, nodes_i, nodes_o, saddles, VF_others


def vortices_evolution(points, epsilon=None):
    """
    Compute the temporal evolution of each vortex centers from a set of points
    at different times. (Points objects must contain only one point)
    Time must be specified in 'v' argument of points.

    Parameters:
    -----------
    pts : tuple of Points objects.
    epsilon : number, optional
        Maximal distance between two successive points.
        default value is Inf.
    """
    if len(points) == 0:
        return None
    if not isinstance(points, ARRAYTYPES):
        raise TypeError("'points' must be an array of Points objects")
    if not isinstance(points[0], Points):
        raise TypeError("'points' must be an array of Points objects")

    # local class to store the point field
    class PointField(object):
        """
        Class representing an orthogonal set of points, defined by a position
        and a time.
        """
        def __init__(self, pts_tupl):
            if not isinstance(pts_tupl, ARRAYTYPES):
                raise TypeError("'pts' must be a tuple of Point objects")
            for pt in pts_tupl:
                if not isinstance(pt, Points):
                    raise TypeError("'pts' must be a tuple of Point objects")
                if not len(pt) == len(pt.v):
                    raise StandardError("v has not the same dimension as xy")
            # if some Points objects contains more than one point, we decompose them
            for i in np.arange(len(pts_tupl)-1, -1, -1):
                if len(pts_tupl[i]) != 1:
                    pts_tupl[i:i+1] = pts_tupl[i].decompose()
            self.points = []
            # possible times determination
            times = []
            for pt in pts_tupl:
                times.append(pt.v[0])
            times = list(sets.Set(times))
            times.sort()
            self.times = times
            for time in times:
                tmp_points = []
                for pt in pts_tupl:
                    if pt.v[0] == time:
                        tmp_points.append(Point(pt.xy[0, 0], pt.xy[0, 1],
                                                pt.v[0]))
                self.points.append(tmp_points)
            self.unit_x = pts_tupl[0].unit_x
            self.unit_y = pts_tupl[0].unit_y
            self.unit_v = pts_tupl[0].unit_v
            self.time_step = np.size(self.points, 0)

#        def __sort_by_time__(self):
#            points_f = []
#            times = []
#            for i in np.arange(self.time_step):
#                times.append(self.points[i][0].v)
#            while True:
#                ind_min = np.argmin(times)
#                if times[ind_min] == 1e99:
#                    break
#                points_f.append(self.points[ind_min])
#                times[ind_min] = 1e99
#            self.points = points_f

        def make_point_useless(self, i, j):
            """
            Make a point of the field useless (None).
            """
            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError()
            self.points[i][j] = None

        def get_points_at_time(self, time):
            """
            Return all the points for a given time.
            """
            if not isinstance(time, int):
                raise TypeError()
#            pts = []
#            for point in self.points[time]:
#                pts.append(point)
#            return pts
            return self.points[time]

    # local class line to store vortex center evolution line
    class Line(object):
        """
        Class representing a line, defined by a set of ordened points.
        """

        def __init__(self, epsilon):
            self.points = []
            self.epsilon = epsilon

        def add_point(self, pts):
            """
            Add a new point to the line.
            """
            if not isinstance(pts, Point):
                raise TypeError("'pts' must be a Point object")
            self.points.append(pts)

        def remove_point(self, ind):
            """
            Remove the designated point of the line.
            """
            if not isinstance(ind, int):
                raise TypeError("'ind' must be an integer")
            if ind < 0 or ind > len(self):
                raise ValueError("'ind' is out of range")
            self.points.pop(ind)

        def choose_starting_point(self, PF):
            """
            Choose in the given field, a new new starting point for a line.
            """
            if not isinstance(PF, PointField):
                raise TypeError()
            start_i = None
            start_j = None
            for i in np.arange(PF.time_step):
                for j in np.arange(len(PF.points[i])):
                    if PF.points[i][j] is not None:
                        start_i = i
                        start_j = j
                        break
                if start_i is not None:
                    break
            if start_i is None:
                return None, None
            self.points.append(PF.points[start_i][start_j])
            PF.make_point_useless(start_i, start_j)
            return start_i, start_j

        def choose_next_point(self, ext_pt_tupl):
            """
            Get the next point of the line (closest one).
            """
            if not isinstance(ext_pt_tupl, ARRAYTYPES):
                raise TypeError()
            if ext_pt_tupl[0] is not None:
                if not isinstance(ext_pt_tupl[0], Point):
                    raise TypeError()
            if len(self.points) == 0:
                raise Warning("there is no starting points")
            # nearest point choice
            dist = []
            for ext_pt in ext_pt_tupl:
                dist.append(distance(ext_pt, self.points[-1]))
            ind_min = np.argmin(dist)
            # if all the points are 'useless', the line should stop
            if dist[ind_min] == 1e99:
                return None
            # else, we check that the point is not too far (using epsilon)
            if epsilon is not None:
                if dist[ind_min] > self.epsilon**2:
                    return None
            # and if not we add the new point
            self.points.append(ext_pt_tupl[ind_min])
            return ind_min

        def export_to_Points(self, PF):
            """
            Export the current line to a Points object.
            """
            xy = []
            v = []
            for pt in self.points:
                xy.append([pt.x, pt.y])
                v.append(pt.v)
            points = Points(xy, v, PF.unit_x, PF.unit_y, PF.unit_v)
            return points

        def display(self):
            """
            Display the line.
            """
            x = []
            y = []
            v = []
            for pt in self.points:
                x.append(pt.x)
                y.append(pt.y)
                v.append(pt.v)
            plt.plot(x, y)
            plt.scatter(x, y, c=v, cmap=plt.cm.hot)

    # local class point to store one point
    class Point(object):
        """
        Class representing a point with a value on it.
        """
        def __init__(self, x, y, v):
            self.x = x
            self.y = y
            self.v = v

    # local function distance to compute distance between Point objects
    def distance(pts1, pts2):
        """
        Compute the distance between two points.
        """
        if pts2 is None or pts1 is None:
            return 1e99
        else:
            return (pts2.x - pts1.x)**2 + (pts2.y - pts1.y)**2
    # Getting the vortex centers trajectory
    PF = PointField(points)
    if len(PF.points) == 0:
        return []
    points_f = []
    while True:
        line = Line(epsilon)
        start_i, start_j = line.choose_starting_point(PF)
        if start_i is None:
            break
        for i in np.arange(start_i + 1, PF.time_step):
            j = line.choose_next_point(PF.get_points_at_time(i))
            if j is None:
                break
            PF.make_point_useless(i, j)
        points_f.append(line.export_to_Points(PF))
    return points_f


def vortices_zoi(velocityfield, windows_size=5, output='vf'):
    """
    For a velocity field, return a set of boxes, defined by axe intervals.
    These boxes represent zones where critical points are present.
    According to the Poincarre-Bendixon theorem.

    Parameters
    ----------
    velocityfield : VelocityField object
    window_size : integer, optional
        Size of the windows returned by the function, in number of field
        points.
    output : string, optional
        If 'vf', tuple of velocityfields are returned,
        If 'ind', tuple of indices intervals are returned.

    Returns
    -------
    VFS : tuple of VelocityFields
        Vector fields representing the arrays of interest
        (contening critical points). An additional argument 'PBI' is
        available for each vector field, indicating the PBI value.
        A PBI of 1 indicate a vortex, a PBI of -1 a saddle point ans a PBI
        of 0 two or more indivising structures.
    """
    if not isinstance(velocityfield, VelocityField):
        raise TypeError("'velocityfield' must be a VelocityField")

    # local fonction for doing x PB sweep
    def make_PBI_sweep(velocityfield, direction):
        """
        Compute the PBI along the given axis, for the given vector field.
        """
        if not isinstance(velocityfield, VelocityField):
            raise TypeError()
        thetas = velocityfield.get_comp('theta').values.filled(0)
        pbi = []
        if direction == 2:
            thetas = np.rot90(thetas, 3)
        prof1 = thetas[:, 0]
        for i in np.arange(thetas.shape[1]):
            # récupération des profiles et concatenation
            prof2 = thetas[0, 0:i]
            prof3 = thetas[:, i]
            prof4 = thetas[-1, 0:i]
            thetas_border = np.array(prof1[::-1])
            thetas_border = np.append(thetas_border, prof2)
            thetas_border = np.append(thetas_border, prof3)
            thetas_border = np.append(thetas_border, prof4[::-1])
            #calcul des différence de theta
            delta_thetas = thetas_border[1::] - thetas_border[0:-1]
            delta_thetas = np.append(delta_thetas,
                                     thetas_border[0] - thetas_border[-1])
            # traitement des points particuliers (passage 0 à 2 pi)
            delta_thetas[delta_thetas > np.pi] -= 2*np.pi
            delta_thetas[delta_thetas < -np.pi] += 2*np.pi
            # Stockage
            pbi.append(np.sum(delta_thetas)/(2.*np.pi))
        if direction == 2:
            pbi = pbi[::-1]
        return np.array(np.round(pbi))

    def check_struc_number(pbi_x, pbi_y):
        """
        Return the possible number of structure in the vector field.
        """
        # finding points of interest (where pbi change)
        poi_x = pbi_x[1::] - pbi_x[0:-1]
        poi_y = pbi_y[1::] - pbi_y[0:-1]
        # computing number of structures
        num_x = len(poi_x[poi_x != 0])
        num_y = len(poi_y[poi_y != 0])
        return num_x, num_y

    def divide_zone(velocityfield, pbi_x, pbi_y):
        """
        Divide the velocityfield in two velocityfields.
        (According to the present structures)
        """
        poi_x = pbi_x[1::] - pbi_x[0:-1]
        poi_y = pbi_y[1::] - pbi_y[0:-1]
        ind_x = np.where(poi_x != 0)[0]
        ind_y = np.where(poi_y != 0)[0]
        if ind_x.shape[0] == 1 and ind_y.shape[0] == 1:
            return None, None
        # finding the longest useless zone to cut the field
        dist_x = np.abs(ind_x[1::] - ind_x[0:-1])
        dist_y = np.abs(ind_y[1::] - ind_y[0:-1])
        if len(dist_x) == 0:
            dist_x = [0]
        if len(dist_y) == 0:
            dist_y = [0]
        if np.max(dist_x) > np.max(dist_y):
            ind_dist_x_max = np.argmax(dist_x)
            # if struct are to intricate, we can't separate them
            if np.abs(ind_x[ind_dist_x_max] - ind_x[ind_dist_x_max + 1])\
                    < windows_size:
                return None, None
            cut_ind = round((ind_x[ind_dist_x_max]
                            + ind_x[ind_dist_x_max + 1])/2.)
            x_min = velocityfield.V.comp_x.axe_x[0]
            x_max = velocityfield.V.comp_x.axe_x[-1]
            cut_position = velocityfield.V.comp_x.axe_x[cut_ind]
            a = velocityfield.trim_area(intervalx=[x_min, cut_position])
            a.theta = velocityfield.theta.trim_area(intervalx=[x_min, cut_position])
            b = velocityfield.trim_area(intervalx=[cut_position, x_max])
            b.theta = velocityfield.theta.trim_area(intervalx=[cut_position, x_max])
            return a, b
        else:
            ind_dist_y_max = np.argmax(dist_y)
            # if struct are to intricate, we can't separate them
            if np.abs(ind_y[ind_dist_y_max] - ind_y[ind_dist_y_max + 1])\
                    < windows_size:
                return None, None
            cut_ind = round((ind_y[ind_dist_y_max] - 0.5
                            + ind_y[ind_dist_y_max + 1] - 0.5)/2)
            y_min = velocityfield.V.comp_x.axe_y[0]
            y_max = velocityfield.V.comp_x.axe_y[-1]
            cut_position = velocityfield.V.comp_x.axe_y[cut_ind]
            a = velocityfield.trim_area(intervaly=[y_min, cut_position])
            a.theta = velocityfield.theta.trim_area(intervaly=[y_min, cut_position])
            b = velocityfield.trim_area(intervaly=[cut_position, y_max])
            b.theta = velocityfield.theta.trim_area(intervaly=[cut_position, y_max])
            return a, b

    def final_triming(velocityfield, pbi_x, pbi_y):
        """
        Trim the velocityfield, according to the windows size.
        """
        demi_ws = round(windows_size*1./2)
        axe_x = velocityfield.V.comp_x.axe_x
        axe_y = velocityfield.V.comp_x.axe_y
        poi_x = pbi_x[1::] - pbi_x[0:-1]
        poi_y = pbi_y[1::] - pbi_y[0:-1]
        # getting the rectangle around points of interest
        if np.all(poi_x == 0) and np.all(poi_y == 0):
            raise ValueError()
        if np.all(poi_x == 0):
            ind_x_min = 0
            ind_x_max = len(poi_x)
        else:
            ind_x_min = np.where(poi_x != 0)[0][0]
            ind_x_max = np.where(poi_x != 0)[0][-1]
        if np.all(poi_y == 0):
            ind_y_min = 0
            ind_y_max = len(poi_y)
        else:
            ind_y_min = np.where(poi_y != 0)[0][0]
            ind_y_max = np.where(poi_y != 0)[0][-1]
        if ind_x_min - demi_ws >= 0:
            x_min = axe_x[ind_x_min - demi_ws]
        else:
            x_min = axe_x[0]
        if ind_x_max + demi_ws <= len(axe_x) - 1:
            x_max = axe_x[ind_x_max + demi_ws]
        else:
            x_max = axe_x[-1]
        if ind_y_min - demi_ws >= 0:
            y_min = axe_y[ind_y_min - demi_ws]
        else:
            y_min = axe_y[0]
        if ind_y_max + demi_ws <= len(axe_y) - 1:
            y_max = axe_y[ind_y_max + demi_ws]
        else:
            y_max = axe_y[-1]
        tmp_vf = velocityfield.trim_area([x_min, x_max], [y_min, y_max])
        return tmp_vf

    def make_steps(velocityfield):
        """
        Divide the velocityfield, and return small velocityfields contening all
        the interesting structures.
        """
        if not isinstance(velocityfield, VelocityField):
            raise TypeError("'velocityfield' must be a VelocityField")
        #removing useless array around critical points
        pbi_x = make_PBI_sweep(velocityfield, 1)
        pbi_y = make_PBI_sweep(velocityfield, 2)
        velocityfield = final_triming(velocityfield, pbi_x, pbi_y)
        vf_pending = [velocityfield]
        vf_treated = []
        while True:
            # condition of breaking the loop
            if len(vf_pending) == 0:
                break
            vf_tmp = vf_pending[0]
            pbi_x = make_PBI_sweep(vf_tmp, 1)
            pbi_y = make_PBI_sweep(vf_tmp, 2)
            nmb_struct = check_struc_number(pbi_x, pbi_y)
            # if there is only one structure in the field
            if np.all(np.array(nmb_struct) == 1):
                # adding PBI value to object
                vf_tmp = final_triming(vf_tmp, pbi_x, pbi_y)
                vf_tmp.PBI = make_PBI_sweep(vf_tmp, 1)[-1]
                vf_treated.append(vf_tmp)
                vf_pending[0:1] = []
                continue
            # if there is no structure of interest in the field
            # (seems useless with te use of final_triming)
            if np.all(nmb_struct == 0):
                vf_pending[0:1] = []
            vf1, vf2 = divide_zone(vf_tmp, pbi_x, pbi_y)
            # if we can't divide again
            if vf1 is None:
                # adding PBI info (non-conclusive)
                vf_tmp = final_triming(vf_tmp, pbi_x, pbi_y)
                vf_tmp.PBI = 0
                vf_treated.append(vf_tmp)
                vf_pending[0:1] = []
                continue
            # we divide the field
            vf_pending[0:1] = []
            vf_pending.append(vf1)
            vf_pending.append(vf2)
        return vf_treated

    def get_linked_indices(vf, velocityfield):
        """
        Return the indices interval on each axis that describe vf
        on velocityfield.
        """
        axe_x, axe_y = velocityfield.get_axes()
        vf_axe_x, vf_axe_y = vf.get_axes()
        xmin = vf_axe_x[0]
        xmax = vf_axe_x[-1]
        ymin = vf_axe_y[0]
        ymax = vf_axe_y[-1]
        ind_x_min = velocityfield.V.comp_x.get_indice_on_axe(1, xmin)[0]
        ind_x_max = velocityfield.V.comp_x.get_indice_on_axe(1, xmax)[-1]
        ind_y_min = velocityfield.V.comp_x.get_indice_on_axe(2, ymin)[0]
        ind_y_max = velocityfield.V.comp_x.get_indice_on_axe(2, ymax)[-1]
        return [ind_x_min, ind_x_max], [ind_y_min, ind_y_max]

    vf_treated = make_steps(velocityfield)
    if output == 'vf':
        return vf_treated
    elif output == 'ind':
        indices = []
        for field in vf_treated:
            indices.append(get_linked_indices(field, velocityfield))
        return indices
    else:
        raise ValueError("Unknown output value : {}".format(output))


def get_sigma(vectorfield, radius=None, ortho=True):
    """
    Return the sigma scalar field, reprensenting the homogeneity of the
    VectorField. Values of 1 mean homogeneous velocity field  and 0 mean
    heterogeneous velocity field. Heterogeneous velocity zone are
    representative of zone with crirical points.
    In details, Sigma is calculated as the variance of the 'ecart' between
    velocity angles of points surrounding the point of interest.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : float
        Radius used to compute sigma. for one value of sigma, the algorithm
        take account of all the points present in a circle of this radius
        around it.
    ortho : Bool, optionnal
        Specify if the field is orthonormed (if axes values are
        homogeneous). If True, the algorithm is faster.

    Returns
    -------
    Sigma : ScalarField
        Sigma scalar field
    """
    if radius is None:
        radius = ((vectorfield.comp_x.axe_x[-1] - vectorfield.comp_x.axe_x[0])
                  / (vectorfield.get_dim()[0] - 1)
                  + (vectorfield.comp_x.axe_y[-1]
                     - vectorfield.comp_x.axe_y[0])
                  / (vectorfield.get_dim()[1] - 1)
                  )/2
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if not ortho:
        raise StandardError("Not implemented yet")
    axe_x, axe_y = vectorfield.get_axes()
    theta = vectorfield.get_theta().values.data
    # Calcul du motif determinant les points voisins
    ptcentral = [axe_x[int(len(axe_x)/2)], axe_y[int(len(axe_y)/2)]]
    motif = np.array(vectorfield.comp_x.get_points_around(ptcentral, radius))
    motif = motif - np.array([int(len(axe_x)/2), int(len(axe_y)/2)])
    nmbpts = len(motif)
    # récupération du masque du champ de vitesse
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        mask = vectorfield.comp_x.values.mask
    else:
        mask = np.zeros(vectorfield.get_dim())
    # Ajout au masque des valeurs sur le bord
    for inds, pos, _ in vectorfield:
        min_x = axe_x[0] + radius
        max_x = axe_x[-1] - radius
        min_y = axe_y[0] + radius
        max_y = axe_y[-1] - radius
        if (pos[0] < min_x or pos[1] < min_y or
                pos[0] > max_x or
                pos[1] > max_y):
            mask[inds[1], inds[0]] = True
    # calcul de delta moyen
    deltamoy = 2.*np.pi/(nmbpts)
    # boucle sur les points
    sigmas = np.zeros(vectorfield.get_dim())
    for inds, pos, _ in vectorfield:
        # On ne fait rien si la valeur est masquée
        if mask[inds[1], inds[0]]:
            continue
        # Extraction des thetas interessants
        theta_nei = np.zeros((len(motif),))
        i = 0
        for indx, indy in motif + inds:
            theta_nei[i] = theta[indy, indx]
            i += 1
        # Tri des thetas et calcul de deltas
        theta_nei.sort()
        delta = np.zeros((len(theta_nei) + 1,))
        for i in np.arange(len(theta_nei) - 1):
            delta[i] = theta_nei[i+1] - theta_nei[i]
        delta[-1] = np.pi*2 - (theta_nei[-1] - theta_nei[0])
        # calcul de sigma
        sigmas[inds[1], inds[0]] = np.var(delta)
    # calcul (analytique) du sigma max
    sigma_max = ((np.pi*2 - deltamoy)**2
                 + (nmbpts - 1)*(0 - deltamoy)**2)/nmbpts
    # normalisation analytique
    sigmas /= sigma_max
    sigmas = np.ma.masked_array(sigmas, mask)
    sigma_sf = ScalarField()
    sigma_sf.import_from_arrays(axe_x, axe_y, sigmas,
                                vectorfield.comp_x.unit_x,
                                vectorfield.comp_x.unit_y, make_unit(""))
    return sigma_sf


def get_gamma1(vectorfield, radius=None, mask=None, sigmafilter=False):
    """
    Return the gamma1 scalar field. Gamma1 criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma1 for each point. If not mentionned, a value is choosen in
        ordre to have about 50 points in the circle. It allow to get good
        result, without big computation cost.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma1 will be compute only where mask is 'False'.
    sigmafilter : boolean, optionnal
        If true, the 'GetSigma' is used before computation of gamma to
        determine the zones of interest. Gamma1 is then only compute in
        these zones.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = ((vectorfield.comp_x.axe_x[-1] - vectorfield.comp_x.axe_x[0])
                  / (vectorfield.get_dim()[0] - 1)
                  + (vectorfield.comp_x.axe_y[-1]
                     - vectorfield.comp_x.axe_y[0])
                  / (vectorfield.get_dim()[1] - 1)
                  )/2
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(sigmafilter, bool):
        raise TypeError("'sigmafilter' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        Vx = vectorfield.comp_x.values.data
        Vy = vectorfield.comp_y.values.data
    else:
        Vx = vectorfield.comp_x.values
        Vy = vectorfield.comp_y.values
    # test d'orthogonalité
    gridx = axe_x[1:-1] - axe_x[0:-2]
    gridy = axe_y[1:-1] - axe_y[0:-2]
    if (all(gridx - gridx[0] < 1e-5*gridx[0])
            and all(gridy - gridy[0] < 1e-5*gridy[0])):
        ortho = True
    else:
        ortho = False
    # application du filtre sigma si nécessaire
    if sigmafilter:
        radius_s = ((axe_x[-1] - axe_x[0])/len(axe_x)*6.
                    + (axe_y[-1] - axe_y[0])/len(axe_y)*6.)/4.
        sigma_sf = vectorfield.get_sigma(radius_s, ortho).values
        mask = np.logical_or(mask, sigma_sf.mask)
        mask = np.logical_or(sigma_sf > .3, mask)
    # récupération du masque sur les vitesses
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        mask = np.logical_or(mask, vectorfield.comp_x.values.mask)
    # suppression des zones de bord à la zone de calcul
    for inds, pos, _ in vectorfield:
        if (pos[0] < axe_x[0] + radius or pos[1] < axe_y[0] + radius or
                pos[0] > axe_x[-1] - radius or
                pos[1] > axe_y[-1] - radius):
            mask[inds[1], inds[0]] = True
    # calcul des normes
    norm_v = np.sqrt(Vx**2 + Vy**2)
    # calcul du motif si on est en ortho
    if ortho:
        ptcentral = [axe_x[int(len(axe_x)/2)], axe_y[int(len(axe_y)/2)]]
        motif = np.array(vectorfield.comp_x.get_points_around(ptcentral,
                                                              radius))
        motif = motif - np.array([int(len(axe_x)/2), int(len(axe_y)/2)])
    # boucle sur les points
    gammas1 = np.zeros(vectorfield.comp_x.get_dim())
    for inds, pos, _ in vectorfield:
        # arrete si zone == False
        if mask[inds[1], inds[0]]:
            continue
        # récupère les points voisins
        if ortho:
            indsaround = motif + np.array(inds)
        else:
            indsaround = vectorfield.comp_x.get_points_around(pos, radius)
        # si il n y a pas de voisins on ne fait rien
        if len(indsaround) == 0:
            continue
        # si il y en a on boucle sur les points voisins
        gamma1 = 0.
        nmbpts = len(indsaround)
        for indaround in indsaround:
            # calcul des vecteurs pour le produit scalaire
            vector_a = [axe_x[indaround[0]] - axe_x[inds[0]],
                        axe_y[indaround[1]] - axe_y[inds[1]]]
            vector_b = [Vx[indaround[1], indaround[0]],
                        Vy[indaround[1], indaround[0]]]
            denom = norm_v[indaround[1],
                           indaround[0]]*((vector_a[0]**2
                                           + vector_a[1]**2)**(1./2))
            if denom == 0:
                pass
            else:
                gamma1 += (vector_a[0]*vector_b[1]
                           - vector_a[1]*vector_b[0])/denom
        gammas1[inds[1], inds[0]] = gamma1/nmbpts
    gammas1 = np.ma.masked_array(gammas1, mask)
    gamma_sf = ScalarField()
    gamma_sf.import_from_arrays(axe_x, axe_y, gammas1,
                                unit_x=vectorfield.comp_x.unit_x,
                                unit_y=vectorfield.comp_x.unit_y)
    return gamma_sf


def get_gamma2(vectorfield, radius=None, mask=None, sigmafilter=False):
    """
    Return the gamma2 scalar field. gamma2 criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma1 for each point. If not mentionned, a value is choosen in
        ordre to have about 50 poins in the circle. It allow to get good
        result, without big computation cost.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        gamma2 will be compute only where mask is 'False'.
    sigmafilter : boolean, optionnal
        If true, the 'GetSigma' is used before computation of gamma to
        determine the zones of interest. gamma2 is then only compute in
        these zones.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = ((vectorfield.comp_x.axe_x[-1] - vectorfield.comp_x.axe_x[0])
                  / (vectorfield.get_dim()[0] - 1)
                  + (vectorfield.comp_x.axe_y[-1]
                     - vectorfield.comp_x.axe_y[0])
                  / (vectorfield.get_dim()[1] - 1)
                  )/2
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(sigmafilter, bool):
        raise TypeError("'sigmafilter' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        Vx = vectorfield.comp_x.values.data
        Vy = vectorfield.comp_y.values.data
    else:
        Vx = vectorfield.comp_x.values
        Vy = vectorfield.comp_y.values
    # test d'orthogonalité
    gridx = axe_x[1:-1] - axe_x[0:-2]
    gridy = axe_y[1:-1] - axe_y[0:-2]
    if (all(gridx - gridx[0] < 1e-5*gridx[0])
            and all(gridy - gridy[0] < 1e-5*gridy[0])):
        ortho = True
    else:
        ortho = False
    # application du filtre sigma si nécessaire
    if sigmafilter:
        radius_s = ((axe_x[-1] - axe_x[0])/len(axe_x)*6.
                    + (axe_y[-1] - axe_y[0])/len(axe_y)*6.)/4.
        sigma_sf = vectorfield.get_sigma(radius_s, ortho).values
        mask = np.logical_or(mask, sigma_sf.mask)
        mask = np.logical_or(sigma_sf > .3, mask)
    # récupération du masque sur les vitesses
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        mask = np.logical_or(mask, vectorfield.comp_x.values.mask)
    # suppression des zones de bord à la zone de calcul
    for inds, pos, _ in vectorfield:
        if (pos[0] < axe_x[0] + radius or pos[1] < axe_y[0] + radius or
                pos[0] > axe_x[-1] - radius or
                pos[1] > axe_y[-1] - radius):
            mask[inds[1], inds[0]] = True
    # calcul du motif si on est en ortho
    if ortho:
        ptcentral = [axe_x[int(len(axe_x)/2)], axe_y[int(len(axe_y)/2)]]
        motif = np.array(vectorfield.comp_x.get_points_around(ptcentral,
                                                              radius))
        motif = motif - np.array([int(len(axe_x)/2), int(len(axe_y)/2)])
    # boucle sur les points
    gammas2 = np.zeros(vectorfield.comp_x.get_dim())
    for inds, pos, _ in vectorfield:
        # arrete si zone == False
        if mask[inds[1], inds[0]]:
            continue
        # récupère les points voisins
        if ortho:
            indsaround = motif + np.array(inds)
        else:
            indsaround = vectorfield.comp_x.get_points_around(pos, radius)
        # si il n y a pas de voisins on ne fait rien
        if len(indsaround) == 0:
            continue
        # si il y en a on calcul la vitesse moyenne sur les voisins
        v_mean = [0., 0.]
        nmbpts = len(indsaround)
        for indaround in indsaround:
            v_mean[0] += Vx[inds[1], inds[0]]
            v_mean[1] += Vy[inds[1], inds[0]]
        v_mean[0] /= nmbpts
        v_mean[1] /= nmbpts
        # puis on boucle sur les poits voisins
        gamma2 = 0.
        for indaround in indsaround:
            # calcul des vecteurs pour le produit scalaire
            A = [axe_x[indaround[0]] - axe_x[inds[0]],
                 axe_y[indaround[1]] - axe_y[inds[1]]]
            B = [Vx[indaround[1], indaround[0]] - v_mean[0],
                 Vy[indaround[1], indaround[0]] - v_mean[1]]
            denom = (((B[0]**2 + B[1]**2)**(1./2))
                     * ((A[0]**2 + A[1]**2)**(1./2)))
            if denom == 0:
                pass
            else:
                gamma2 += (A[0]*B[1] - A[1]*B[0])/denom
        gammas2[inds[1], inds[0]] = gamma2/nmbpts
    gammas2 = np.ma.masked_array(gammas2, mask)
    gamma_sf = ScalarField()
    gamma_sf.import_from_arrays(axe_x, axe_y, gammas2,
                                unit_x=vectorfield.comp_x.unit_x,
                                unit_y=vectorfield.comp_x.unit_y)
    return gamma_sf


def get_kappa1(vectorfield, radius=None, mask=None, sigmafilter=False):
    """
    Return the kappa1 scalar field. kappa1 criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma1 for each point. If not mentionned, a value is choosen in
        ordre to have about 50 poins in the circle. It allow to get good
        result, without big computation cost.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        kappa1 will be compute only where mask is 'False'.
    sigmafilter : boolean, optionnal
        If true, the 'GetSigma' is used before computation of kappa to
        determine the zones of interest. kappa1 is then only compute in
        these zones.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = ((vectorfield.comp_x.axe_x[-1] - vectorfield.comp_x.axe_x[0])
                  / (vectorfield.get_dim()[0] - 1)
                  + (vectorfield.comp_x.axe_y[-1]
                     - vectorfield.comp_x.axe_y[0])
                  / (vectorfield.get_dim()[1] - 1)
                  )/2
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(sigmafilter, bool):
        raise TypeError("'sigmafilter' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        Vx = vectorfield.comp_x.values.data
        Vy = vectorfield.comp_y.values.data
    else:
        Vx = vectorfield.comp_x.values
        Vy = vectorfield.comp_y.values
    # test d'orthogonalité
    gridx = axe_x[1:-1] - axe_x[0:-2]
    gridy = axe_y[1:-1] - axe_y[0:-2]
    if (all(gridx - gridx[0] < 1e-5*gridx[0])
            and all(gridy - gridy[0] < 1e-5*gridy[0])):
        ortho = True
    else:
        ortho = False
    # application du filtre sigma si nécessaire
    if sigmafilter:
        radius_s = ((axe_x[-1] - axe_x[0])/len(axe_x)*6.
                    + (axe_y[-1] - axe_y[0])/len(axe_y)*6.)/4.
        sigma_sf = vectorfield.get_sigma(radius_s, ortho).values
        mask = np.logical_or(mask, sigma_sf.mask)
        mask = np.logical_or(sigma_sf > .3, mask)
    # récupération du masque sur les vitesses
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        mask = np.logical_or(mask, vectorfield.comp_x.values.mask)
    # suppression des zones de bord à la zone de calcul
    for inds, pos, _ in vectorfield:
        if (pos[0] < axe_x[0] + radius or pos[1] < axe_y[0] + radius or
                pos[0] > axe_x[-1] - radius or
                pos[1] > axe_y[-1] - radius):
            mask[inds[1], inds[0]] = True
    # calcul des normes
    norm_v = np.sqrt(Vx**2 + Vy**2)
    # calcul du motif si on est en ortho
    if ortho:
        ptcentral = [axe_x[int(len(axe_x)/2)], axe_y[int(len(axe_y)/2)]]
        motif = np.array(vectorfield.comp_x.get_points_around(ptcentral,
                                                              radius))
        motif = motif - np.array([int(len(axe_x)/2), int(len(axe_y)/2)])
    # boucle sur les points
    kappas1 = np.zeros(vectorfield.comp_x.get_dim())
    for inds, pos, _ in vectorfield:
        # arrete si zone == False
        if mask[inds[1], inds[0]]:
            continue
        # récupère les points voisins
        if ortho:
            indsaround = motif + np.array(inds)
        else:
            indsaround = vectorfield.comp_x.get_points_around(pos, radius)
        # si il n y a pas de voisins on ne fait rien
        if len(indsaround) == 0:
            continue
        # si il y en a on boucle sur les points voisins
        kappa1 = 0.
        nmbpts = len(indsaround)
        for indaround in indsaround:
            # calcul des vecteurs pour le produit scalaire
            A = [axe_x[indaround[0]] - axe_x[inds[0]],
                 axe_y[indaround[1]] - axe_y[inds[1]]]
            B = [Vx[indaround[1], indaround[0]],
                 Vy[indaround[1], indaround[0]]]
            denom = norm_v[indaround[1],
                           indaround[0]]*((A[0]**2 + A[1]**2)**(1./2))
            if denom == 0:
                pass
            else:
                kappa1 += (A[0]*B[0] + A[1]*B[1])/denom
        kappas1[inds[1], inds[0]] = kappa1/nmbpts
    kappas1 = np.ma.masked_array(kappas1, mask)
    kappa_sf = ScalarField()
    kappa_sf.import_from_arrays(axe_x, axe_y, kappas1,
                                unit_x=vectorfield.comp_x.unit_x,
                                unit_y=vectorfield.comp_x.unit_y)
    return kappa_sf


def get_kappa2(vectorfield, radius=None, mask=None, sigmafilter=False):
    """
    Return the kappa2 scalar field. kappa2 criterion is used in
    vortex analysis.
    The fonction recognize if the field is ortogonal, and use an
    apropriate algorithm.

    Parameters
    ----------
    vectorfield : VectorField object
    radius : number, optionnal
        The radius used to choose the zone where to compute
        gamma1 for each point. If not mentionned, a value is choosen in
        ordre to have about 50 poins in the circle. It allow to get good
        result, without big computation cost.
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        kappa2 will be compute only where mask is 'False'.
    sigmafilter : boolean, optionnal
        If true, the 'GetSigma' is used before computation of kappa to
        determine the zones of interest. kappa2 is then only compute in
        these zones.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if radius is None:
        radius = ((vectorfield.comp_x.axe_x[-1] - vectorfield.comp_x.axe_x[0])
                  / (vectorfield.get_dim()[0] - 1)
                  + (vectorfield.comp_x.axe_y[-1]
                     - vectorfield.comp_x.axe_y[0])
                  / (vectorfield.get_dim()[1] - 1)
                  )/2
    if not isinstance(radius, NUMBERTYPES):
        raise TypeError("'radius' must be a number")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'zone' must be an array of boolean")
    else:
        mask = np.array(mask)
    if not isinstance(sigmafilter, bool):
        raise TypeError("'sigmafilter' must be a boolean")
    axe_x, axe_y = vectorfield.get_axes()
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        Vx = vectorfield.comp_x.values.data
        Vy = vectorfield.comp_y.values.data
    else:
        Vx = vectorfield.comp_x.values
        Vy = vectorfield.comp_y.values
    # test d'orthogonalité
    gridx = axe_x[1:-1] - axe_x[0:-2]
    gridy = axe_y[1:-1] - axe_y[0:-2]
    if (all(gridx - gridx[0] < 1e-5*gridx[0])
            and all(gridy - gridy[0] < 1e-5*gridy[0])):
        ortho = True
    else:
        ortho = False
    # application du filtre sigma si nécessaire
    if sigmafilter:
        radius_s = ((axe_x[-1] - axe_x[0])/len(axe_x)*6.
                    + (axe_y[-1] - axe_y[0])/len(axe_y)*6.)/4.
        sigma_sf = vectorfield.get_sigma(radius_s, ortho).values
        mask = np.logical_or(mask, sigma_sf.mask)
        mask = np.logical_or(sigma_sf > .3, mask)
    # récupération du masque sur les vitesses
    if isinstance(vectorfield.comp_x.values, np.ma.MaskedArray):
        mask = np.logical_or(mask, vectorfield.comp_x.values.mask)
    # suppression des zones de bord à la zone de calcul
    for inds, pos, _ in vectorfield:
        if (pos[0] < axe_x[0] + radius or pos[1] < axe_y[0] + radius or
                pos[0] > axe_x[-1] - radius or
                pos[1] > axe_y[-1] - radius):
            mask[inds[1], inds[0]] = True
    # calcul du motif si on est en ortho
    if ortho:
        ptcentral = [axe_x[int(len(axe_x)/2)], axe_y[int(len(axe_y)/2)]]
        motif = np.array(vectorfield.comp_x.get_points_around(ptcentral,
                                                              radius))
        motif = motif - np.array([int(len(axe_x)/2), int(len(axe_y)/2)])
    # boucle sur les points
    kappas1 = np.zeros(vectorfield.comp_x.get_dim())
    for inds, pos, _ in vectorfield:
        # arrete si zone == False
        if mask[inds[1], inds[0]]:
            continue
        # récupère les points voisins
        if ortho:
            indsaround = motif + np.array(inds)
        else:
            indsaround = vectorfield.comp_x.get_points_around(pos, radius)
        # si il n y a pas de voisins on ne fait rien
        if len(indsaround) == 0:
            continue
        # si il y en a on calcul la vitesse moyenne sur les voisins
        v_mean = [0., 0.]
        nmbpts = len(indsaround)
        for indaround in indsaround:
            v_mean[0] += Vx[inds[1], inds[0]]
            v_mean[1] += Vy[inds[1], inds[0]]
        v_mean[0] /= nmbpts
        v_mean[1] /= nmbpts
        # puis on boucle sur les poits voisins
        kappa2 = 0.
        for indaround in indsaround:
            # calcul des vecteurs pour le produit scalaire
            A = [axe_x[indaround[0]] - axe_x[inds[0]],
                 axe_y[indaround[1]] - axe_y[inds[1]]]
            B = [Vx[indaround[1], indaround[0]] - v_mean[0],
                 Vy[indaround[1], indaround[0]] - v_mean[1]]
            denom = (((B[0]**2 + B[1]**2)**(1./2))
                     * ((A[0]**2 + A[1]**2)**(1./2)))
            if denom == 0:
                pass
            else:
                kappa2 += (A[0]*B[0] + A[1]*B[1])/denom
        kappas1[inds[1], inds[0]] = kappa2/nmbpts
    kappas1 = np.ma.masked_array(kappas1, mask)
    kappa_sf = ScalarField()
    kappa_sf.import_from_arrays(axe_x, axe_y, kappas1,
                                unit_x=vectorfield.comp_x.unit_x,
                                unit_y=vectorfield.comp_x.unit_y)
    return kappa_sf


def get_q_criterion(vectorfield, mask=None):
    """
    Return the normalized scalar field of the 2D Q criterion .
    Define as "1/2*(sig**2 - S**2)" , with "sig" the deformation tensor,
    norm and "S" the rate of rotation tensor norm.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optional
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.get_axes()
    Vx = vectorfield.comp_x.values.copy()
    Vy = vectorfield.comp_y.values.copy()
    #calcul des gradients
    Exy, Exx = np.gradient(Vx)
    Eyy, Eyx = np.gradient(Vy)
    # calcul de 'sig**2' et 'S**2'
    sig = 2.*(Exy - Eyx)**2
    S = (2.*Exx)**2 + 2.*(Eyx + Exy)**2 + (2.*Eyy)**2
    # calcul de Qcrit
    qcrit = 1./2*(sig - S)
    qcrit = np.ma.masked_array(qcrit, mask)
    qcrit = qcrit/np.abs(qcrit).max()
    q_sf = ScalarField()
    q_sf.import_from_arrays(axe_x, axe_y, qcrit,
                            unit_x=vectorfield.comp_x.unit_x,
                            unit_y=vectorfield.comp_x.unit_y)
    return q_sf


def get_iota(vectorfield, mask=None):
    """
    Return the iota scalar field. iota criterion is used in
    vortex analysis.
    The fonction is only usable on orthogonal fields.
    Warning : This function is minimum at the saddle point center, and
    maximum around this point.

    Parameters
    ----------
    vectorfield : VectorField object
    mask : array of boolean, optionnal
        Has to be an array of the same size of the vector field object,
        iota2 will be compute only where zone is 'False'.
    """
    if not isinstance(vectorfield, VectorField):
        raise TypeError("'vectorfield' must be a VectorField object")
    if mask is None:
        mask = np.zeros(vectorfield.get_dim())
    elif not isinstance(mask, ARRAYTYPES):
        raise TypeError("'mask' must be an array of boolean")
    else:
        mask = np.array(mask)
    axe_x, axe_y = vectorfield.get_axes()
    # récupération de theta et de son masque
    theta = vectorfield.get_theta().values
    if isinstance(theta, np.ma.MaskedArray):
        mask = np.logical_or(mask, theta.mask)
        theta = theta.data
    # récupération des dx et dy
    dx = np.abs(axe_x[0] - axe_x[2])
    dy = np.abs(axe_y[0] - axe_y[2])
    # boucle sur les points
    grad_theta_m = np.zeros(vectorfield.get_dim())
    maskf = mask.copy()
    for inds, _, _ in vectorfield:
        # arrete si on est sur un bords ou sur une valeurs masquée
        if (inds[1] == 0 or inds[0] == 0 or inds[1] == len(axe_y)-1
                or inds[0] == len(axe_x)-1):
            maskf[inds[1], inds[0]] = True
            continue
        if np.any(mask[inds[1]-1:inds[1]+2, inds[0]-1:inds[0]+2]):
            maskf[inds[1], inds[0]] = True
            continue
        # calcul de theta_m autours
        theta_m = [[theta[inds[1] - 1, inds[0] - 1] + 3./4*np.pi,
                   theta[inds[1], inds[0] - 1] + np.pi/2,
                   theta[inds[1] + 1, inds[0] - 1] + 1./4*np.pi],
                  [theta[inds[1] - 1, inds[0]] + np.pi,
                   0.,
                   theta[inds[1] + 1, inds[0]]],
                  [theta[inds[1] - 1, inds[0] + 1] - 3./4*np.pi,
                   theta[inds[1], inds[0] + 1] - np.pi/2,
                   theta[inds[1] + 1, inds[0] + 1] - np.pi/4]]
        theta_m = np.mod(theta_m, np.pi*2)
        grad_x = (theta_m[2, 1] - theta_m[0, 1])/dx
        grad_y = (theta_m[1, 2] - theta_m[1, 0])/dy
        # calcul de gradthetaM
        grad_theta_m[inds[1], inds[0]] = (grad_x**2 + grad_y**2)**(1./2)
    # application du masque
    grad_theta_m = np.ma.masked_array(grad_theta_m, maskf)
    iota_sf = ScalarField()
    iota_sf.import_from_arrays(axe_x, axe_y, grad_theta_m,
                               unit_x=vectorfield.comp_x.unit_x,
                               unit_y=vectorfield.comp_x.unit_y)
    ### XXX : Alternate calculation for iota !
    mean = iota_sf.integrate_over_surface().asNumber()
    mean = mean/len(iota_sf.axe_x)/len(iota_sf.axe_y)
    iota_sf = iota_sf - mean
    iota_sf = iota_sf**2
    ###
    return iota_sf
