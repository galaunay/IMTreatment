# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:21:52 2014

@author: muahah
"""

def get_bl_thickness(profile, perc=0.9):
    """
    Return the profile boundary layer thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object
    perc : float, optionnal
        Percentage used in the bl calculation (90% per default).

    Returns
    -------
    BLT : float
        Boundary layer thickness, in axe x unit.
    """
    value = profile.get_interpolated_value(y=profile.get_max()*perc)
    return value[0]*profile.unit_x


def get_displ_thickness(profile):
    """
    Return the profile displacement thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object
    """
    # adding a x(0) value
    if profile.x[0] != 0:
        pos_x = np.append([0], profile.x)
        pos_y = np.append([0], profile.y)
    else:
        pos_x = profile.x
        pos_y = profile.y
    fonct = 1 - np.abs(pos_y)/np.max(np.abs(pos_y))
    delta = np.trapz(fonct, pos_x)
    return delta*profile.unit_x


def get_momentum_thickness(profile):
    """
    Return the profile momentum thickness.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object
    """
    # adding a x(0) value
    if profile.x[0] != 0:
        pos_x = np.append([0], profile.x)
        pos_y = np.append([0], profile.y)
    else:
        pos_x = profile.x
        pos_y = profile.y
    fonct = np.abs(pos_y)/np.max(np.abs(pos_y))*(1 - np.abs(pos_y)
                                                 / np.max(np.abs(pos_y)))
    delta = np.trapz(fonct, pos_x)
    return delta*profile.unit_x


def get_shape_factor(profile):
    """
    Return the profile shape factor.
    WARNING : the wall must be at x=0.

    Parameters
    ----------
    profile : Profile object
    """
    shape_factor = profile.get_displ_thickness()\
        / profile.get_momentum_thickness()
    return shape_factor
