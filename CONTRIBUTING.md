# Contributing to Medical Insurance Cost Prediction

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards others

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide detailed description**
- **Explain why this enhancement would be useful**
- **List examples** of how it would work

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/medical-insurance-prediction.git
cd medical-insurance-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 pylint
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Single quotes for strings, double for docstrings

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black src/ web_app/

# Check formatting without making changes
black --check src/
```

### Linting

We use **flake8** for linting:

```bash
# Run flake8
flake8 src/ web_app/

# With specific configuration
flake8 --max-line-length=100 --ignore=E203,W503 src/
```

### Type Hints

Use type hints for function signatures:

```python
def process_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Process the dataframe."""
    # Implementation
    return X, y
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        
    Returns:
        Trained model object
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Implementation
```

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Keep commits atomic (one logical change per commit)
- Add tests for new features
- Update documentation

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# Check coverage
pytest --cov=src tests/

# Run linting
flake8 src/
black --check src/
```

### 4. Commit Guidelines

Follow conventional commits:

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Format code
refactor: Refactor code
test: Add tests
chore: Update dependencies
```

Example:
```bash
git commit -m "feat: Add XGBoost hyperparameter tuning"
git commit -m "fix: Correct BMI validation in web app"
git commit -m "docs: Update README with deployment instructions"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** following commit conventions
- **Description** of changes
- **Related issues** (if any)
- **Screenshots** (if UI changes)
- **Checklist** completed

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
- [ ] Dependent changes merged

## Testing Guidelines

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_preprocessing.py
import pytest
from src.preprocessing import DataPreprocessor

def test_missing_value_handling():
    """Test missing value handling."""
    # Setup
    df = create_test_dataframe()
    preprocessor = DataPreprocessor(df)
    
    # Execute
    result = preprocessor.handle_missing_values(strategy='mean')
    
    # Assert
    assert result.isnull().sum().sum() == 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

### Test Coverage

Aim for:
- **Minimum**: 70% coverage
- **Target**: 85% coverage
- **Critical paths**: 100% coverage

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include type hints
- Provide usage examples
- Document exceptions

### README Updates

Update README.md if you:
- Add new features
- Change installation process
- Modify usage instructions
- Add dependencies

### Documentation Files

Update relevant docs:
- `DOCUMENTATION.md` - Technical documentation
- `QUICKSTART.md` - Quick start guide
- `CONTRIBUTING.md` - This file

## Project Structure

When adding new files, follow the structure:

```
src/
  ├── module_name.py      # Main module
  └── __init__.py         # Package init

tests/
  └── test_module_name.py # Tests for module

web_app/
  ├── app.py              # Flask app
  ├── templates/          # HTML templates
  └── static/             # CSS, JS, images
```

## Code Review Process

### For Contributors

- Respond to feedback promptly
- Make requested changes
- Ask questions if unclear
- Be patient and respectful

### For Reviewers

- Be constructive and kind
- Explain reasoning
- Suggest improvements
- Approve when ready

## Release Process

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to PyPI (maintainers only)

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Chat**: Join our Discord (if available)
- **Email**: Contact maintainers

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in documentation

## Additional Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Pytest Documentation](https://docs.pytest.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

Thank you for contributing! 🚀

**Questions?** Feel free to ask in Issues or Discussions.
