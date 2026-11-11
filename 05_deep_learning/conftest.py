"""Skip the TensorFlow appendix when the optional `tf` extra is not installed."""

from importlib.util import find_spec

collect_ignore_glob = [] if find_spec("tensorflow") else ["appendix/*"]
