#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension

setup(
    name="cython_gsl",
    version="0.1a",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/CythonGSL",
    packages=["cython_gsl"],
    package_data={"cython_gsl":["*.pxi"]},
    description="""Cython wrapper for the Gnu Scientific Library.""",
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
