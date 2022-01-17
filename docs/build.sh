
python setup.py build
export PYTHONPATH="$(ls -d $(pwd)/build/lib*/):$PYTHONPATH"

# sphinx-quickstart
cd docs
sphinx-apidoc -o ./source ../ffrecord
make clean && make html
