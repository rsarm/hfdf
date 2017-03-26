## HFDF
### Calculate forces using the Helmann-Feynman  from the basis set expansion of the density with PySCF.

This does a density fitting with PySCF using the functions given in the
tutorial page http://sunqm.github.io/pyscf/tutorial.html#access-ao-integrals and
with the fitted density computes forces using the Hellmann-Feynmann theorem.

### Compiling PySCF and adding the 'cint1e_drinv_sph' integral.

The thing is that one needs the integral (\| nabla-rinv \|) for the calculation of the forces with the Hellmann-Feynmann theorem.
For that it can be used the clisp auto-intor.cl utility that comes with pyscf.
This is done as follows:
(See section 'Generating integrals' in https://github.com/sunqm/libcint)


* Download the code
* cd pyscf/lib/; mkdir build; cd build; cmake ..;

* Now this part is nasty: Run 'make' and stop it (crtl+c) after libcint was cloned.

* cd deps/src/libcint/scripts/    # This is inside pyscf/lib/build

* Add the line   '("cint1e_drinv_sph"        spheric  (\| nabla-rinv \|)) in the block that reads:
```python
(gen-cint "grad1.c"
  '("cint1e_ipovlp_sph"       spheric  (nabla \|))
  '("cint1e_ipkin_sph"        spheric  (.5 nabla \| p dot p))
  '("cint1e_ipnuc_sph"        spheric  (nabla \| nuc \|))
  '("cint1e_iprinv_sph"       spheric  (nabla \| rinv \|))
  '("cint1e_rinv_sph"         spheric  (\| rinv \|))
  '("cint2e_ip1_sph"          spheric  (nabla \, \| \,))
)
```
   so now it reads as
```python
(gen-cint "grad1.c"
  '("cint1e_ipovlp_sph"       spheric  (nabla \|))
  '("cint1e_ipkin_sph"        spheric  (.5 nabla \| p dot p))
  '("cint1e_ipnuc_sph"        spheric  (nabla \| nuc \|))
  '("cint1e_iprinv_sph"       spheric  (nabla \| rinv \|))
  '("cint1e_drinv_sph"        spheric  (\| nabla-rinv \|))
  '("cint1e_rinv_sph"         spheric  (\| rinv \|))
  '("cint2e_ip1_sph"          spheric  (nabla \, \| \,))
)
```
* Now run
   clisp auto_intor.cl
   mv *.c ../src/autocode/

* Now go back to pyscf/lib/build and run 'make' this time
   until it finishes. That's it!

* Finaly add the full pyscf directory to the PATH
