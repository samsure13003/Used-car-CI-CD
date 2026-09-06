import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        filename = "unknown"
        line_number = "unknown"
    else:
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

    return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        filename, line_number, str(error)
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message_detail(error_message, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
