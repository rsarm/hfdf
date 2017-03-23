
## Calculate Helmann-Feynman forces from the basis set expansion of the density with PySCF.



### Compiling PySCF and adding the 'cint1e_drinv_sph' integral.

The thing is that one needs the integral (\| nabla-rinv \|) for the calculations.
For that it can be used the clisp auto-intor.cl utility that comes with pyscf.
This is done as follows:
(See section 'Generating integrals' in https://github.com/sunqm/libcint)


1. Download the code
2. cd pyscf/lib/; mkdir build; cd build; cmake ..;

3. Now this part is nasty:
4. Run 'make' and stop it (crtl+c) after libcint was cloned.

5. cd deps/src/libcint/scripts/    # This is inside pyscf/lib/build

6. Add the line   '("cint1e_drinv_sph"        spheric  (\| nabla-rinv \|)) 
   in the block that reads:
(gen-cint "grad1.c"
  '("cint1e_ipovlp_sph"       spheric  (nabla \|))
  '("cint1e_ipkin_sph"        spheric  (.5 nabla \| p dot p))
  '("cint1e_ipnuc_sph"        spheric  (nabla \| nuc \|))
  '("cint1e_iprinv_sph"       spheric  (nabla \| rinv \|))
  '("cint1e_rinv_sph"         spheric  (\| rinv \|))
  '("cint2e_ip1_sph"          spheric  (nabla \, \| \,))
)
   so now it reads as
(gen-cint "grad1.c"
  '("cint1e_ipovlp_sph"       spheric  (nabla \|))
  '("cint1e_ipkin_sph"        spheric  (.5 nabla \| p dot p))
  '("cint1e_ipnuc_sph"        spheric  (nabla \| nuc \|))
  '("cint1e_iprinv_sph"       spheric  (nabla \| rinv \|))
  '("cint1e_drinv_sph"        spheric  (\| nabla-rinv \|))
  '("cint1e_rinv_sph"         spheric  (\| rinv \|))
  '("cint2e_ip1_sph"          spheric  (nabla \, \| \,))
)

7. Now run
   clisp auto_intor.cl
   mv *.c ../src/autocode/

8. Now go back to pyscf/lib/build and run 'make' this time
   until it finishes. That's it!

9. Finaly add the full pyscf directory to the PATH
