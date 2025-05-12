# Contributing to AI-Driven Early Detection of Autism in Toddlers

Thank you for your interest in contributing to our project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### 1. Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/healthcare.git
   cd healthcare
   ```
3. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

### 2. Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-fix-name
   ```

2. Make your changes following our coding standards:
   - Follow PEP 8 style guide
   - Write docstrings for all functions and classes
   - Add type hints
   - Write unit tests for new features
   - Update documentation as needed

3. Run tests and linting:
   ```bash
   pytest
   flake8
   mypy
   ```

4. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

### 3. Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're adding new features
3. The PR will be reviewed by maintainers
4. Address any review comments
5. Once approved, your PR will be merged

## Development Guidelines

### Code Style

- Use Python 3.8+ features
- Follow PEP 8 style guide
- Use type hints
- Write meaningful variable and function names
- Keep functions small and focused
- Add comments for complex logic

### Documentation

- Update README.md for major changes
- Add docstrings to all functions and classes
- Update relevant documentation files
- Include examples for new features

### Testing

- Write unit tests for new features
- Ensure all tests pass
- Maintain or improve test coverage
- Include edge cases in tests

### Model Development

When contributing to model development:

1. Document model architecture
2. Include performance metrics
3. Add validation results
4. Provide example usage
5. Include model cards

### Data Processing

When working with data:

1. Document data sources
2. Include preprocessing steps
3. Add data validation
4. Document feature engineering
5. Include data quality checks

## Project Structure

```
healthcare/
├── src/
│   ├── models/          # Model implementations
│   ├── utils/           # Utility functions
│   └── tests/           # Test files
├── notebooks/           # Jupyter notebooks
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Getting Help

- Open an issue for bugs
- Use discussions for questions
- Join our community forum
- Contact maintainers

## Recognition

Contributors will be:
- Listed in the README.md
- Acknowledged in release notes
- Given credit in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 