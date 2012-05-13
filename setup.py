#!/usr/bin/env python
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

setup(
    name="CythonGSL",
    version="0.2.1alpha",
    author="Thomas V. Wiecki",
    author_email="thomas.wiecki@gmail.com",
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
