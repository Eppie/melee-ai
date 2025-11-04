# Nano-Melee

This is a project for training a model to play Super Smash Bros. Melee.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd nano-melee
    ```

2.  Create a virtual environment and install the dependencies.

    Using `uv`:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

    Using `venv` and `pip`:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### Running Tests

To run the tests, run the following command from the root of the project:

```bash
pytest
```
