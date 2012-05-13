#!/usr/bin/env python
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from glob import glob
import os.path
import cython_gsl
from os.path import splitext

ext_modules = []
for pyxfile in glob(os.path.join('cython_gsl', 'test', '*.pyx')):
    ext_name = splitext(os.path.split(pyxfile)[-1])[0]
    ext = Extension(ext_name,
                    [pyxfile],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    cython_include_dirs=[cython_gsl.get_cython_include_dir()])

    ext_modules.append(ext)

setup(
    name="CythonGSL_test",
    version="0.2",
    author="Thomas V. Wiecki",
    author_email="thomas.wiecki@gmail.com",
    url="http://github.com/twiecki/CythonGSL",
    packages=["cython_gsl.test"],
    package_data={"cython_gsl.test": ["*.py"]},
    description="""Cython declarations for the Gnu Scientific Library.""",
    setup_requires=['Cython', 'CythonGSL'],
    install_requires=['Cython', 'CythonGSL'],
    classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
