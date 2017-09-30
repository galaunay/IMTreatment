#!/bin/bash

cd docs/
sphinx-apidoc -e -f -o . '../IMTreatment/'
make html
