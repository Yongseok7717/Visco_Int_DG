Hello, I am a FEniCS user interested in *coninuous/discontinuous Galerkin finite element methods*.  
I have contributed to code implementations for solving **viscoelastic problems**.  
My works forcused on numerical solutions to **dynamic viscoelastic problems**. 
Code implementations deal with simple wave equations, elastic problems, vicoelastic problems modelled by *generalised Maxwell solid* or *fractional order model*.  
Numerical schemes are based on *spatial finite element methods* (CG/DG) and *Crank-Nicolson method* for time discretisation.


In this numerical experiments, we consider DG FEM for dynamic linear viscoelasticity model with internal variables.

For more details of the model problem, we refer to the autour's [PhD thesis (Chapter 4)](https://bura.brunel.ac.uk/handle/2438/21084).

All codes are constructed in Python 3.6.9 and FEniCS(dolfin) 2019.1.0.
- **out_S1_matrix.py**: Solve the viscoelasticity problem with the displacement form.
- **out_S2_matrix.py**: Solve the viscoelasticity problem with the velocity form.
- **graphic_linear.py** and **graphic_quad.py**: Illustrate graphs of numerical errors on *log-log* scale when $h\approx\Delta t$.
- **main.sh**: Main task to run (consider both examples with linear polinomial basis as well as quadratic).


If you have any inquires, please contact me at email yongseok.jang@cerfacs.fr or yongseok20007717@gmail.com.
If you are interested in my research, please visit https://yongseokmath.wordpress.com/.
