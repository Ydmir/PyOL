init:
	pip install -r requirements.txt

test:
	nosetests tests

clean:
	rm -rf build
	rm -rf dist
	rm -rf pyol.egg-info
	rm -rf __pycache__
