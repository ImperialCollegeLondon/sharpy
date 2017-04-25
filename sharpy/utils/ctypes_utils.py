import ctypes as ct
import platform


def import_ctypes_lib(route, libname):
    lib_path = route + libname
    if platform.system() == 'Darwin':
        ext = '.dylib'
    elif platform.system() == 'Linux':
        ext = '.so'
    else:
        raise NotImplementedError('The platform ' + platform.system() + 'is not supported')

    lib_path += ext
    try:
        library = ct.CDLL(lib_path, mode=ct.RTLD_GLOBAL)
    except:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    return library
