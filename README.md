# SmartCer
**SmartCer** is a Gymnasium environment for the simulation of Renewable Energy Communities (CERs). SmartCer allows the
development and testing of reinforcement learning algorithms for the control of energy storage systems in a CER. The
objective of these RL-based controllers is to minimize energy costs, maximize incentives, and maintain grid balance.  
The environment is based on the [Gymnasium](https://gymnasium.farama.org/index.html) framework, which provides a
standardized interface for reinforcement learning environments.  

For detailed information on how to set up and use the environment, refer to the sections below.

---

## Requirements

- Python ≥ 3.10
- Virtual Python environment
- Python package manager (pip)

---

## Installation

1. Clone the repository locally:
    ```bash
    git clone https://github.com/unisi-lab305/SmartCer
    cd SmartCer
    ```
2. Create and activate a Python virtual environment:
    - Using `venv`:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Using `conda`:
        ```bash
        conda create --name smartcer python=3.10
        conda activate smartcer
        ```
3. Install the required dependencies:
    - For standard installation:
        ```bash
        pip install .
        ```
    - For editable installation (for development):
        ```bash
        pip install -e .
        ```

---

## Usage

Here’s a basic example of how to interact with **SmartCer**:

```python
# TODO
```

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./LICENSE) file.

