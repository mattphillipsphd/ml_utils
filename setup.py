from setuptools import setup, find_packages
from distutils.util import convert_path

version = None
with open("general/__init__.py") as fp:
    for line in fp:
        if line.startswith("__version__"):
            N = len(line)
            version = line[ line.index("\"")+1 : N-line[::-1].index("\"")-1 ]
            break

PROJECT_DIRS = ["general", ]

if __name__ == '__main__':
    setup(setup_fpath=__file__,
        name="general_ml_utils",
        author="Matt Phillips, Kitware",
        author_email="matt.phillips@kitware.com",
        version=version,
        project_dirs=PROJECT_DIRS
    )

