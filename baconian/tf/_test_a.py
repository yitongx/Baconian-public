from tf_parameters import ParametersWithTensorflowVariable
import unittest
from baconian.config.global_config import GlobalConfig

config = GlobalConfig()
print(config.DEFAULT_LOGGING_FORMAT)
config.DEFAULT_LOGGING_FORMAT = "%(message)s"
print(config.DEFAULT_LOGGING_FORMAT)
