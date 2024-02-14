# Slimplectic Integration -- JAX Edition

## TOOD

- [x] Make the names consistent in `Solver`
- [x] Check why I have a different fractional energy error (DHO notebook, last graph)
  - Boom it works!
- [ ] Test in 2D system
- [ ] Test in time dependent system
- [x] Make consistent the treatment of `t`, atm I am having to manually fudge everything
  - I think this is done? 
  - Should it work with `t` with variable time steps?
- [ ] Write written documentation for what lagrangian, k_potential, q0, pi0, and the end results all mean.
- [x] Write in code documentation for the Solver
- [ ] Speed test on my M2 Mac and on the GPU box
  - Does it work with JAX on Metal?

### PostNewtonian

- [ ] Issues with the final graphs
- [ ] Also why is the RK result different???
