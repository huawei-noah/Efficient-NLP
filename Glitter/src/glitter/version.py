"""
Verbatim from: https://github.com/allenai/allennlp/blob/main/allennlp/version.py
"""
_MAJOR = "1"
_MINOR = "0"
# On master and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "0"

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}".format(_MAJOR, _MINOR, _PATCH)
