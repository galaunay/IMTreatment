#!/bin/bash

cd docs/
SPHINX_APIDOC_OPTIONS=members sphinx-apidoc -e -f -o . '../IMTreatment/'
sphinx-build -b coverage . _build/coverage/
sphinx-apidoc -e -f -o . '../IMTreatment/'
more _build/coverage/python.txt
