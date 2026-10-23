.PHONY: env test test-slow lint gpu-test

env:
	python -m pip install -r requirements.txt

test:
	python -m pytest -m "not slow and not gpu" -q

test-slow:
	python -m pytest -m "slow and not gpu" -q

lint:
	python -m ruff check .

gpu-test:
	@echo "Run on Kaggle (GPU session), then paste the log into the PR:"
	@echo '  python -m pytest -m gpu -q'
