#!/bin/bash

# bumpversion minor
sudo python3 -m pip install --upgrade setuptools wheel twine
sudo rm -rf dist
sudo python3 setup.py sdist bdist_wheel
twine upload dist/*
# git push
# git push --tags
