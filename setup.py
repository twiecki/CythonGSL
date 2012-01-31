#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
import os.path
import Cython

cython_dir = os.path.split(Cython.__file__)[0]
cython_include_dir = os.path.join(cython_dir, 'Includes')

print "Install cython-gsl like this:\n python setup.py build\n python setup.py install\n python setup.py install --install-lib=%s" % cython_include_dir

setup(
    name="CythonGSL",
    version="0.1a",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/CythonGSL",
    packages=["cython_gsl"],
    package_data={"cython_gsl":["*.pxi", "*.pxd"]},
    description="""Cython declarations for the Gnu Scientific Library.""",
    classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ]
)
