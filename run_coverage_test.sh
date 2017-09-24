echo "=== Running tests and generating html ==="
pytest --mpl --cov-report html:cov_html --cov IMTreatment tests/
echo "=== Opening html report ==="
firefox tests/cov_html/index.html
