from baconian.common.logging import ConsoleLogger
from baconian.config.global_config import GlobalConfig
import os
ConsoleLogger().init(to_file_flag=GlobalConfig().DEFAULT_WRITE_CONSOLE_LOG_TO_FILE_FLAG,
                         to_file_name=os.path.join(GlobalConfig().DEFAULT_LOG_PATH,
                                                   GlobalConfig().DEFAULT_CONSOLE_LOG_FILE_NAME),
                         level=GlobalConfig().DEFAULT_LOG_LEVEL,
                         logger_name=GlobalConfig().DEFAULT_CONSOLE_LOGGER_NAME)
ConsoleLogger().print('info', 'test')