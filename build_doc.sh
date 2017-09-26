#!/bin/bash

cd docs/
sphinx-apidoc -f -o . '../IMTreatment/'
make html
