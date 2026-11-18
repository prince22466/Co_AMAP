"""
Module implementing generic communication patterns with Python in / Python out
supporting many (nested) primitives + special data science types like DataFrames
or np.ndarrays, with gRPC + protobuf as a backing implementation.
"""

import os
import sys

# Compile the protobuf / gRPC stubs against the local runtime so the gateway
# and inference server can run on different docker-python image versions without
# version-mismatch errors.
import kaggle_evaluation.core.proto_compiler as _proto_compiler

_proto_compiler.ensure_compiled()

module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)


__version__ = '1.8.6'


def help():
    """Print the kaggle_evaluation user guide."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path) as f:
            print(f.read())
    else:
        print('kaggle_evaluation user guide not found.')
        print('See the competition page for submission instructions.')
