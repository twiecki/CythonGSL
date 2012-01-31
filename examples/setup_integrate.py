from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import os
import cython_gsl

setup(
    name="integrate",
    version="0.1",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/CythonGSL",
    description="CythonGSL example integrate.",
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    classifiers=[
                'Development Status :: 3 - Alpha',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ],

    ext_modules = [Extension("integrate",
                             ["integrate.pyx"],
                             libraries=['gsl','gslcblas'],
                             library_dirs=cython_gsl.get_library_dir(),
                             cython_include_dirs=cython_gsl.get_cython_include_dir())]
)

