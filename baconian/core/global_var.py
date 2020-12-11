_global_obj_arg_dict = {}
_global_name_dict = {}

assert id(_global_obj_arg_dict) == id(globals()['_global_obj_arg_dict']) == id(locals()['_global_obj_arg_dict'])
assert id(_global_name_dict) == id(globals()['_global_name_dict']) == id(locals()['_global_name_dict'])


def reset_all():
    globals()['_global_obj_arg_dict'] = {}
    globals()['_global_name_dict'] = {}


def reset(key: str):
    globals()[key] = {}


def get_all() -> dict:
    return dict(
        _global_obj_arg_dict=globals()['_global_obj_arg_dict'],
        _global_name_dict=globals()['_global_name_dict']
    )


class Singleton(object):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        pass


class SinglentonStepCounter(Singleton):
    __instance = None
    __is_first_init = True
    _val = 0
    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, init_val=0):
        if self.__is_first_init:
            self._val = init_val
            self.__is_first_init = False

    def increase(self, increment=None):
        self._val += 1 if not increment else increment
        assert self._val >= 0

    def decrease(self, decrement=None):
        self._val -= 1 if not decrement else decrement
        assert self._val >= 0

    @property
    def val(self):
        return self._val