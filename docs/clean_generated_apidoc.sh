#!/bin/bash
echo "Cleaning auto-generated api doc"
sed -i 's/IMTreatment\\.core\\.points module/Points class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.profile module/Profile class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.vectorfield module/VectorField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.scalarfield module/ScalarField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.temporalscalarfield module/TemporalScalarField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.temporalvectorfield module/TemporalVectorField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.spatialscalarfield module/SpatialScalarField class/g' IMTreatment.*.rst
sed -i 's/IMTreatment\\.core\\.spatialvectorfield module/SpatialVectorField class/g' IMTreatment.*.rst
echo "Done"
