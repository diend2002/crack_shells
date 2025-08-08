# Physics-Informed Neural Networks for Shell Structures

This repository contains the code for solving shell structure problems using Physics-Informed Neural Networks (PINNs). The implementation is based on the paper "Physics-Informed Neural Networks for shell structures".

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- SciPy

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pleizsonoob/crack_embedding_shell.git
    cd crack_embedding_shell
    ```
2.  Install the required packages:
    ```bash
    pip install torch numpy matplotlib scipy
    ```

## How to Run

The main script to run the simulation is `main.py`. You can select the study to run by changing the `study` variable in the `main.py` file.

### Selecting a Study

Open `main.py` and change the `study` variable to one of the following values:

-   `'hyperb_parab'`: Hyperbolic paraboloid shell with a clamped side.
-   `'scordelis_lo'`: Scordelis-Lo roof.
-   `'hemisphere'`: Hemisphere shell with a concentrated load.

```python
if __name__ == '__main__':
    
    # select study: ['hyperb_parab', 'scordelis_lo', 'hemisphere']
    study = 'scordelis_lo'
```

### Running the Simulation

To run the simulation, execute the `main.py` script:

```bash
python main.py
```

The script will perform the following steps:

1.  **Set up parameters**: Loads the parameters for the selected study from `params.py`.
2.  **Sample collocation points**: Generates collocation points for training the PINN.
3.  **Initialize the model**: Initializes the PINN model (`PINN` or `PyKAN_PINN`).
4.  **Train the model**: Trains the PINN using the Adam and L-BFGS optimizers.
5.  **Evaluate the model**: Computes the L2 error and saves the loss history.
6.  **Plot results**: Plots the predicted shell deformation and compares it with the FEM solution.

### Output Files

The script will generate the following output files:

-   `kan_models/`: Directory to save the trained models.
-   `eval/`: Directory to save the evaluation results.
-   `loss_history/`: Directory to save the loss and error history.
-   `models/pn_statedict.pt`: The state dictionary of the trained model.

## Code Structure

-   `main.py`: The main script to run the simulation.
-   `params.py`: Contains the parameters for different studies.
-   `src/`: Directory containing the source code.
    -   `pinn_model.py`: Defines the PINN model.
    -   `pinnkan_model.py`: Defines the PINN model using the PyKAN library.
    -   `geometry.py`: Defines the shell geometry.
    -   `shell_model.py`: Defines the shell model (Linear Naghdi-Koiter).
    -   `material_model.py`: Defines the material model (Linear Elastic).
    -   `eval_utils.py`: Contains utility functions for evaluation and plotting.
    -   `utils.py`: Contains utility functions for the simulation.
    -   `crack_geometry.py`: Defines the crack geometry.

## Adding a New Study

To add a new study, you need to:

1.  Add a new entry in the `create_param_dict` function in `params.py`.
2.  Define the geometry of the new study in the `get_mapping` function in `src/geometry.py`.
3.  Provide the corresponding FEM solution file for comparison.
