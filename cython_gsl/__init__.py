def get_include():
    import sys, os

    if sys.platform == "win32":
        # Hardcoded paths under windows :-/
        gsl_include = r"c:\Program Files\GnuWin32\include"
    else:
        gsl_include = os.popen('gsl-config --cflags').read()[2:-1]

    assert gsl_include != '', "Couldn't find gsl. Make sure it's installed and in the path."

    return gsl_include

def get_library_dir():
    import sys, os

    if sys.platform == "win32":
        # Hardcoded paths under windows :-/
        lib_gsl_dir = r"c:\Program Files\GnuWin32\lib"
    else:
        lib_gsl_dir = ''

    return [lib_gsl_dir]

def get_cython_include_dir():
    import cython_gsl
    return [cython_gsl.__path__]

