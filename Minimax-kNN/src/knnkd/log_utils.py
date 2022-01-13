import logging
import re


def set_global_logging_error(prefixes=None):
    set_global_logging_level(logging.ERROR, prefixes)


def set_global_logging_warning(prefixes=None):
    set_global_logging_level(logging.WARNING, prefixes)


def set_global_logging_info(prefixes=None):
    set_global_logging_level(logging.INFO, prefixes)


def set_global_logging_level(level=logging.ERROR, prefixes=None):
    """
    Taken from: https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    if prefixes is None:
        prefixes = [""]

    prefix_re = re.compile(fr'^(?:{ "|".join(prefixes) })')

    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)