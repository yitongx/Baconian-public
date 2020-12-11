from baconian.core.status import StatusWithSingleInfo
from baconian.core.core import Basic

class Experiment(Basic):
    def __init__(self):
        self._status = StatusWithSingleInfo(obj=self)
        # print(self._status)
        print('self: ', self)

    def get_status(self) -> dict:
        print('get_status:')
        return self()

exp = Experiment()
exp.set_status('name')
exp.set_status('id')
print(exp())