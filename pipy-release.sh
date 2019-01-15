#!/bin/bash

sudo rm -rf dist
sudo python3 setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
