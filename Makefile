.PHONY: install test clean build build-dist publish

install:
	pip install -e .

test:
	pytest

clean:
	find embedden -name '*.pyc' -exec rm -f {} +
	find embedden -name '*.pyo' -exec rm -f {} +
	find embedden -name '*~' -exec rm -f  {} +
	find embedden -name '*.so' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/

build:
	python setup.py build_ext --inplace

build-dist: clean
	python setup.py sdist
	python setup.py bdist_wheel

publish: 
	twine upload dist/*
