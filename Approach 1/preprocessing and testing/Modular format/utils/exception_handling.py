"""
Custom exception class with file + line context for easier debugging.
"""

import sys


def _error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error in [{file_name}] at line [{line_number}]: {error}"
    )


class CustomException(Exception):
    """Wraps any exception with file name and line number context."""

    def __init__(self, error: Exception, error_detail: sys = sys):
        super().__init__(str(error))
        self.error_message = _error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
