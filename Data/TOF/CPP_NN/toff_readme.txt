### Purpose of the provided program 'Toff' ###

This C++ program reads in the five tables that contain the ToF-coefficients up to the 7th order.
The program provides access to the coeffcients through five functions that can be called 
by the user as described in Step 4 of this readme file. The code has been tested to run 
under various Microsoft Visual Studio versions and g++ compilers on Linux systems.  


### Step 1: installing the program 'Toff' ###

- Copy the directory 'Toff' in a directory <your_dir> of your choice, or leave it where it is. 

-The 'Toff' folder contains 2 folders. Folder 'InputData' shall contain the five read-in 
tables, which are provided in the 'TABLES' folder upon download of the TOF7-package from 
the repository. The folder 'CSource' contains the C++ source code. 

-Copy the five asci tables into the 'InputData' folder.


### Step 2: creating an executable file 'toff.exe' ###

LINUX users:
-requirements: the 'g++' or the 'c++' compiler
-open a shell window and change into 'your_dir/Toff'
-type ./toff_script.txt

If successful: 
4 new files appear in the Toff-folder: 'toff.o' and 'toff.exe', and two text files 
with possible error messages or warnings. The executable file 'toff.exe' can be run 
by typing './toff.exe' in the shell.

Errors: 
If the 'toff.o' file has not been created, a compilation error must have occurred.
See the file 'toff_compile.txt' for error messages.
If only the 'toff.exe' file has not been created, a linkage error must have occurred.
See the file 'toff_link.txt' for error messages.

WINDOWS users:
-requirements: VS Studio OR cygwin/gcc. In the latter case, procede as LINUX users.
-create a new 'Project' ('Empty project') and add it to a 'Solution'.
-Add all files from the 'CSource' directory to the project except for 'toff_config.cpp'.
Do not add the latter, or exclude it from Build.


### Step 3: run the program ###

-run the executable file 'toff.exe'. An error message will occur: 
('could not open file '/your_dir/Toff/InputData/...)
-open the file 'CSource/toff_globals.cpp' and adjust the absolute directory path, which
is stored in the variable 'g_absDir'.
-run the script again (compile & link)
-run the .exe-file again. If the message 'END of program' is reached, the tables were read 
successfully.



### Step 4: using the coefficients ###

The program provides five functions to access the ToF7-coefficients:

 double coeff_S(int k, int n, double *sn);
 double coeff_Sprime(int k, int n, double *sn);
 double coeff_m(int k, double *sn);
 double inner_f(int n, double* sn);
 double inner_fprime(int n, double* sn);

where 

'k' is even whole number (type int)
'n' is even whole number (type int)
'sn' is array of floating point numbers (type double) of size 0+1, where 'O' is the desired 
order of ToF-approximation, e.g. O=7. This array is supposed to contain the local values of
the figure fuctions s2,s4,...,s14 at radial coordinate value l. 

Function 'coeff_S(k,n,sn)' returns the sum of coefficients c_ink over index i,
see Section A.3 in Ref. [1]. 

Function 'coeff_Sprime(k,n,sn)' returns the sum of coefficients c'_ink over index i,
see Section A.3 in Ref. [1]. 

Function 'coeff_m(k,sn)' returns the sum of coefficients c_i0k over index i. 
To obtain the local value of A^(Q)_k this sum needs to be multiplied by m_rot, 
see Section A.1 and Eq. (A1) in Ref [1]. 

Function 'inner_f(n,sn)' returns the sum of coefficients c_in over index i, 
which yields the local value of fn(l), see Section A.3 and Eq. (A9) in Ref. [1].

Function 'inner_fprime(n,sn)' returns the sum of coefficients c'_in over index i, 
which yields the local value of f'n(l), see Section A.3 and Eq. (A9) in Ref. [1].


An example for how these functions can be called is provided in the file 'toff_main.cpp'.



### References ###

[1] Nettelmann N. and Movshovitz M. and Ni D. et al, Planetary Science J. (2021), under review.
