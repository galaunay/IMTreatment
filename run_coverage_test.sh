echo "=== Running tests ==="
coverage run --source="IMTreatment" -m unittest discover tests
echo "=== Generating html report ==="
coverage html
echo "=== Opening html report ==="
firefox htmlcov/index.html
