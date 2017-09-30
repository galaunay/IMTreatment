#!/bin/bash

cd docs/
sphinx-apidoc -e -f -o . '../IMTreatment/'
bash clean_generated_apidoc.sh
make html
