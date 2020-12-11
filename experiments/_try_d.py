from baconian.core.global_var import SinglentonStepCounter
from _try_c import Func

counter = SinglentonStepCounter()
counter.increase()
print(counter.val)
print(type(counter))

counter2 = Func().call()
counter2.increase()
print(counter2.val)
print(type(counter2))
print(id(counter) == id(counter2))