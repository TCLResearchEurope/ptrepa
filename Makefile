MODULE_NAME=prepa

PY_DIRS=src/ptrepa tests setup.py

PY_MYPY_FLAKE8=src/ptrepa tests setup.py

FILES_TO_CLEAN=src/ptrepa.egg-info dist

include Makefile.inc
