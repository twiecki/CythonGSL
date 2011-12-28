CythonGSL
*********

CythonGSL provides a Cython interface for the GNU Scientific Library (GSL).

Cython is the ideal tool to speed up numerical computations. While SciPy provides many numerical tools such as optimizers and integrators they still are not as fast as they could be because of the Python function call overhead (even if your evaluation function is written cython). CythonGSL allows you to shave off that last layer by calling directly into the c functions from your Cython code.

Fork of PyrexGsl by Mario Pernici (http://wwwteor.mi.infn.it/~pernici/pyrexgsl/pyrexgsl.html).

Dependencies
************

* Python
* Cython
* GSL (for a windows port see http://gnuwin32.sourceforge.net/packages/gsl.htm)

Usage
*****

In your cython file from which you want to call gsl routines, import gsl:
>>> include "CythonGSL/gsl.pxi"

From there you should be able to call any gsl function, see http://www.gnu.org/software/gsl/manual/gsl-ref.html for the GSL reference.

Compilation
***********

I haven't quite figured out the best way to find the gsl directories in the setup.py file under windows, any suggestions are welcome.

Here is what a setup.py could look like:

::

    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    import numpy as np
    import os
    import sys
    print sys.platform

    if sys.platform == "win32":
        # Hardcoded paths under windows :-/
        gsl_include = r"c:\Program Files\GnuWin32\include"
        lib_gsl_dir = r"c:\Program Files\GnuWin32\lib"
    else:
        gsl_include = os.popen('gsl-config --cflags').read()[2:-1]
        lib_gsl_dir = ''

    if gsl_include == '':
        print "Couldn't find gsl. Make sure it's installed and in the path."
        sys.exit(-1)

    setup(
        [...]
        include_dirs = [np.get_include(), gsl_include],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("my_cython_script", ["src/my_cython_script.pyx"], libraries=['gsl','gslcblas'], library_dirs=[lib_gsl_dir])]
        )

