#!/bin/bash

# Numerical error tables and figures
date
python3 out_S1_matrix.py -a 10 -b 1 -k 1 -i 1 -I 8 -j 1 -J 8|& tee -a S1_linear.txt

date
python3 out_S2_matrix.py -a 10 -b 1 -k 1 -i 1 -I 8 -j 1 -J 8|& tee -a S2_linear.txt

date
python3 out_S1_matrix.py -a 10 -b 1 -k 2 -i 1 -I 8 -j 1 -J 8|& tee -a S1_quadratic.txt

date
python3 out_S2_matrix.py -a 10 -b 1 -k 2 -i 1 -I 8 -j 1 -J 8|& tee -a S2_quadratic.txt

date
python3 graphic_linear.py
python3 graphic_quad.py



# Numerical orders
date
python3 out_S1_matrix.py -a 10 -b 1 -k 1 -i 2 -I 6 -j 11 -J 12|& tee -a order_S1_linear.txt

date
python3 out_S2_matrix.py -a 10 -b 1 -k 1 -i 2 -I 6 -j 11 -J 12|& tee -a order_S2_linear.txt

date
python3 out_S1_matrix.py -a 10 -b 1 -k 2 -i 2 -I 6 -j 11 -J 12|& tee -a order_S1_quadratic.txt

date
python3 out_S2_matrix.py -a 10 -b 1 -k 2 -i 2 -I 6 -j 11 -J 12|& tee -a order_S2_quadratic.txt

date
python3 out_S1_matrix.py -a 10 -b 1 -k 2 -i 7 -I 8 -j 1 -J 5|& tee -a timeorder_S1_quadratic.txt

date
python3 out_S2_matrix.py -a 10 -b 1 -k 2 -i 7 -I 8 -j 1 -J 5|& tee -a timeorder_S2_quadratic.txt
