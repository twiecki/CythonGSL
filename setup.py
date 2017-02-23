#!/usr/bin/env python
from distutils.core import setup

setup(
    name="CythonGSL",
    version="0.2.2",
    author="Thomas V. Wiecki",
    author_email="thomas.wiecki@gmail.com",
    url="http://github.com/twiecki/CythonGSL",
    packages=["cython_gsl"],
    package_data={"cython_gsl": ["*.pxd"]},
    description="""Cython declarations for the Gnu Scientific Library.""",
    setup_requires=['Cython >= 0.16'],
    install_requires=['Cython >= 0.16'],
    classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering']
)
