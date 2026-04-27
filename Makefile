.PHONY: test test-e2e test-parallel test-cov lint

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run E2E tests only
test-e2e:
	pytest tests/test_full_workflow.py tests/test_ev_recovery_e2e.py tests/test_admin_api.py -v

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	pytest tests/ -n auto -v

# Run with coverage
test-cov:
	pytest tests/ --cov=app --cov-report=html --cov-report=term --cov-fail-under=90

# Lint
lint:
	# ruff check app/ tests/ # ruff might not be installed
	# mypy app/
	echo "Linting skipped (install ruff/mypy to enable)"

# Start test database
test-db-up:
	docker-compose -f tests/docker-compose.test.yml up -d

# Stop test database
test-db-down:
	docker-compose -f tests/docker-compose.test.yml down -v

# Full CI pipeline
ci: test-cov
