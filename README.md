# Oil Displacement Using Polymer
This project is a 1D reservoir simulator that models oil displacement through water and polymer injection. It uses the IMPES (Implicit Pressure Explicit Saturation) method to solve the equations in porous media.
The standout of this project is the high-performance given by the _JAX_ library, which enables _JIT_ (Just-In-Time) compilation and Automatic Differentiation for economic optimization.

## Module Structure
- `impes/input.json` : Configuration file containing reservoir and injection parameters
- `src/solvers/impes_numpy.py` : The first version of the simulator, implemented in _NumPy_ using the IMPES method to determine pressure, saturation, and polymer concentration, also calculates cumulative oil production.
- `src/solvers/impes_jax.py` : Optimized version of the simulator written in JAX. which utilizes all of _JAX_ features and functions resulting in much better performance.
- `utils/plots.py` : Module responsible for generating the result plots. It plots the profiles along the reservoir for:
  - Water Saturation ($S_w$)
  - Pressure ($P$)
  - Polymer Concentration ($C$)
  - Cumulative Oil Production
- `main.py`  : Main execution script. It calls the desired version of the simulator (_NumPy_ or _JAX_), save the returned _pandas_ `DataFrames` in `.txt` files in `outputs/` folder and calls the plot archive
- `optimization.py` : Economic optimization script that uses the `minimize` routine from _SciPy_ library combined with `jax.value_and_grad` function and `jax.jit` for better performance. The goal is is to find the optimal injection time that maximizes the profit function
## Requisites
To run the simulator, you will need Python 3 installed along with the following libraries:
  ```bash
  pip install numpy scipy pandas matplotlib jax jaxlib

  ```
(Obs: To get maximum performance from JAX, setting up the environment according to the __official JAX documentation__ is recommended if using a GPU)

## How to Run
The simulator has two workflows: Physical Study and Economic Optimization. 

1. **Physical Study**
   If you want to run the simulation for specific injection times and visualize the physical and production profiles:
   
   1. Edit the reservoir and injection parameters in `impes/input.json` if necessary
   2. In the `main.py` choose between _JAX_ or _NumPy_ by changing the `mode` variable
   3. Execute the main script:
      ```bash
      python main.py
      ```
    **Outputs** : The script will generate _pandas_ `DataFrames` converted in `.txt` files in the `outputs/` folder and open _Matplotlib_ windows showing Saturation, Pressure, Concentration, and Production.

2. **Economic Optimization**
   If you want to find the optimal polymer injection time:
   
   1. Execute the optimization script:
      ```bash
      python optimization.py
      ```
   **Outputs** : The terminal outputs the function value and gradient at each step, and in the end displays a plot comparing the minimize routine with a brute-force search algorithm.

## Additional Notes
 - For further details on the problem, access the project report: https://www.overleaf.com/read/mtvnnmvfttgy#d59639
    
  

 
