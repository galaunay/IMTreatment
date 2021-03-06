#!/bin/bash

cd docs/

# ============================================
# Clean
# ============================================
rm -rf _build

# ============================================
# Generate autodoc
# ============================================
sphinx-apidoc -M -f -e --ext-autodoc --ext-viewcode -o . "../IMTreatment"

# ============================================
# Clean autogenerated doc
# ============================================
printf "Cleaning auto-generated api doc... "
# Classes
sed -i 's/IMTreatment.core.points module/Points class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.profile module/Profile class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.field module/Field class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.vectorfield module/VectorField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.scalarfield module/ScalarField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.temporalfields module/TemporalFields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.fields module/Fields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.temporalscalarfields module/TemporalScalarFields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.temporalvectorfields module/TemporalVectorFields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.spatialfields module/SpatialFields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.spatialscalarfields module/SpatialScalarFields class/g' IMTreatment.*.rst
sed -i 's/IMTreatment.core.spatialvectorfields module/SpatialVectorFields class/g' IMTreatment.*.rst
# Modules
sed -i 's/IMTreatment.\([^.]*\).\1 module/\1 module/g' IMTreatment.*.rst
sed -i 's/IMTreatment.\([^.]*\).\([^.]*\) module/\1.\2 module/g' IMTreatment.*.rst
# Packages
sed -i 's/IMTreatment.\([^.]*\) package/\1 package/g' IMTreatment.*.rst
sed -i 's/Submodules\n----------//g' IMTreatment.*.rst
sed -i '/^Submodules/ { N ; N ; d }' IMTreatment.*.rst
printf "Done\n"

# ============================================
# Make doc with sphinx
# ============================================
make html
