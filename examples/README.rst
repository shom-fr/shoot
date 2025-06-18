.. _examples:

Examples
========

This gallery shows examples that integrate several aspects of shoot.
This examples works in a sequential way. Mind that however things 
are implicitly parallelized through xarray methods. 


Parallelisation  
---------------

The eddy detection can be parallelized easily.
In laptops change the paral boolean argument to True in Eddies.detect_eddies 
and in EvolEddies.detect_eddies. 
To run with large grid on HPC, wrap your example in if __name__ == "__main__"
and add the following lines : 

import multiprocessing as mp \\
mp.set_start_method("spawn")


