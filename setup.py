#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
import os.path
import Cython

setup(
    name="CythonGSL",
    version="0.1.1",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/CythonGSL",
    packages=["cython_gsl"],
    package_data={"cython_gsl":["*.pxd"]},
    description="""Cython declarations for the Gnu Scientific Library.""",
    setup_requires=['Cython'],
    install_requires=['Cython'],
    classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering']
)
