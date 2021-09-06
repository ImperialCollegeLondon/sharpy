import ctypes as ct
import platform
import os


def import_ctypes_lib(route, libname):
    # lib_path = route + libname
    lib_path = os.path.join(route, libname)
    if platform.system() == 'Darwin':
        ext = '.dylib'
    elif platform.system() == 'Linux':
        ext = '.so'
    else:
        raise NotImplementedError('The platform ' + platform.system() + 'is not supported')

    lib_path += ext
    lib_path = os.path.abspath(lib_path)
    try:
        library = ct.CDLL(lib_path, mode=ct.RTLD_GLOBAL)
    except:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    return library
