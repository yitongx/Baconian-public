import os
from baconian.core.global_var import SinglentonStepCounter

class Func(object):
    def __init__(self):
        self.counter = SinglentonStepCounter()
    def call(self, *args, **kwargs):
        return self.counter