# Copyright (c) 2025-, Georgia Institute of Technology.
# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

"""Utilities for reporting dolfinx version info."""

import os
import sys

# Avoid shadowing stdlib "warnings" when running this file directly from utils/.
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _entry in ("", _script_dir):
    while _entry in sys.path:
        sys.path.remove(_entry)

import dolfinx


def dlversion():
    """
    Return the dolfinx version as a tuple (major, minor, patch).
    """
    version_str = getattr(dolfinx, "__version__", "0.0.0")
    vv = [ss for ss in version_str.split(".")]
    vv += ["0"] * (3 - len(vv))
    return (int(vv[0]), int(vv[1]), int(vv[2]))


# Only FEniCSx 0.10.x is supported
supported_versions = [(0, 10)]


def checkdlversion():
    """
    Check if :code:`FEniCSx` version is supported.
    """
    ver = dlversion()
    major_minor = (ver[0], ver[1])
    if major_minor not in supported_versions:
        print(f"The version of FEniCSx ({ver[0]}.{ver[1]}.{ver[2]}) you are using is not supported.")
        print("hippylibx requires FEniCSx 0.10.x")
        exit()
    print(f"FEniCSx version: {ver[0]}.{ver[1]}.{ver[2]}")
