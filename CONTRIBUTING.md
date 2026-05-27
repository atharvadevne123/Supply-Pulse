# Contributing to Supply-Pulse

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Supply-Pulse
cd Supply-Pulse
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy pre-commit
pre-commit install
```

## Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Run `make lint && make test`
5. Submit a pull request

## Code Standards

- **Style**: ruff with `E,F,W,I` rules, line length 100
- **Tests**: pytest — maintain ≥70% coverage
- **Types**: add type annotations to all new functions
- **Docs**: Google-style docstrings for all public APIs

## Commit Message Format

```
type(scope): short description

Types: feat, fix, test, ci, docs, chore, refactor, perf
```

## Running Tests

```bash
make test
# or
pytest tests/ -v --cov=app
```
