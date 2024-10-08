# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import atexit
import logging
import sys
import traceback

from ..log import get_module_logger
from . import R
from .recorder import Recorder

logger = get_module_logger("workflow", logging.INFO)


# function to handle the experiment when unusual program ending occurs
def experiment_exit_handler():
    """
    Method for handling the experiment when any unusual program ending occurs.
    The `atexit` handler should be put in the last, since, as long as the program ends, it will be called.
    Thus, if any exception or user interruption occurs beforehand, we should handle them first. Once `R` is
    ended, another call of `R.end_exp` will not take effect.

    Limitations:
    - If pdb is used in your program, excepthook will not be triggered when it ends.  The status will be finished
    """
    sys.excepthook = experiment_exception_hook  # handle uncaught exception
    atexit.register(R.end_exp, recorder_status=Recorder.STATUS_FI)  # will not take effect if experiment ends


def experiment_exception_hook(exc_type, value, tb):
    """
    End an experiment with status to be "FAILED". This exception tries to catch those uncaught exception
    and end the experiment automatically.

    Parameters
    exc_type: Exception type
    value: Exception's value
    tb: Exception's traceback
    """
    logger.error(f"An exception has been raised[{exc_type.__name__}: {value}].")

    # Same as original format
    traceback.print_tb(tb)
    print(f"{exc_type.__name__}: {value}")

    R.end_exp(recorder_status=Recorder.STATUS_FA)
