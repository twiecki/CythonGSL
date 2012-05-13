from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import os
import cython_gsl
import numpy as np
from glob import glob
from os.path import splitext

ext_modules = []
for pyxfile in glob('*.pyx'):
    ext = Extension(splitext(pyxfile)[0],
                    [pyxfile],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    cython_include_dirs=[cython_gsl.get_cython_include_dir()])

    ext_modules.append(ext)

setup(
    name="CythonGSL_examples",
    version="0.2",
    author="Thomas V. Wiecki",
    author_email="thomas.wiecki@gmail.com",
    url="http://github.com/twiecki/CythonGSL",
    description="CythonGSL examples.",
    include_dirs = [np.get_include(), cython_gsl.get_include()],
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
    ext_modules = ext_modules
)
