from abc import ABCMeta, abstractmethod
# import sharpy.utils.cout_utils as cout
import os

class cout(object):
    def cout_wrap(arg, arg2):
        print(arg)

dict_of_lc = {}
lc = {}  # for internal working


# decorator
def lagrangeconstraint(arg):
    # global available_solvers
    global dict_of_lc
    try:
        arg._lc_id
    except AttributeError:
        raise AttributeError('Class defined as lagrange constraint has no _lc_id attribute')
    dict_of_lc[arg._lc_id] = arg
    return arg

def print_available_lc():
    cout.cout_wrap('The available lagrange constraints on this session are:', 2)
    for name, i_lc in dict_of_lc.items():
        cout.cout_wrap('%s ' % i_lc._lc_id, 2)

def lc_from_string(string):
    return dict_of_lc[string]

def lc_list_from_path(cwd):
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if ".py" in onlyfiles[i_file]:
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_lc(lc_name, print_info=True):
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % lc_name, 2)
    cls_type = lc_from_string(lc_name)
    lc = cls_type()
    return lc


class BaseLagrangeConstraint(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_n_eq(self):
        pass

    @abstractmethod
    def initialise(self, **kwargs):
        pass

    @abstractmethod
    def staticmat(self, **kwargs):
        pass

    @abstractmethod
    def dynamicmat(self, **kwargs):
        pass

    @abstractmethod
    def staticpost(self, **kwargs):
        pass

    @abstractmethod
    def dynamicpost(self, **kwargs):
        pass



@lagrangeconstraint
class SampleLagrange(BaseLagrangeConstraint):
    _lc_id = 'SampleLagrange'

    def __init__(self):
        self._n_eq = 3

    def get_n_eq(self):
        return self._n_eq

    def initialise(self, **kwargs):
        print('Type of LC: ', self._lc_id)
        print('Arguments and values:')
        for k, v in kwargs.items():
            print(k, v)

        return

    def staticmat(self, **kwargs):
        return np.zeros((6, 6))

    def dynamicmat(self, **kwargs):
        return np.zeros((10, 10))

    def staticpost(self, **kwargs):
        return

    def dynamicpost(self, **kwargs):
        return

# this at the end of the file
print_available_lc()

# test
if __name__ == '__main__':
    lc_list = list()
    lc_list.append(lc_from_string('SampleLagrange')())
    lc_list.append(lc_from_string('SampleLagrange')())

    counter = -1
    for lc in lc_list:
        counter += 1
        lc.initialise(counter=counter)





















