#    CythonGSL provides a set of Cython declarations for the GNU Scientific Library (GSL).
#    Copyright (C) 2012 Thomas V. Wiecki
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

def get_include():
    import sys, os

    if sys.platform == "win32":
        gsl_include = os.getenv('LIB_GSL')
        if gsl_include is None:
            # Environmental variable LIB_GSL not set, use hardcoded path.
            gsl_include = r"c:\Program Files\GnuWin32\include"
        else:
            gsl_include += "/include"
    else:
        gsl_include = os.popen('gsl-config --cflags').read()[2:-1]

    assert gsl_include != '', "Couldn't find gsl. Make sure it's installed and in the path."

    return gsl_include

def get_library_dir():
    import sys, os

    if sys.platform == "win32":
        lib_gsl_dir = os.getenv('LIB_GSL')
        if lib_gsl_dir is None:
            # Environmental variable LIB_GSL not set, use hardcoded path.
            lib_gsl_dir = r"c:\Program Files\GnuWin32\lib"
        else:
            lib_gsl_dir += "/lib"
    else:
        lib_gsl_dir = os.popen('gsl-config --libs').read().split()[0][2:]

    return lib_gsl_dir

def get_libraries():
    return ['gsl', 'gslcblas']

def get_cython_include_dir():
    import cython_gsl, os.path
    return os.path.split(cython_gsl.__path__[0])[0]

