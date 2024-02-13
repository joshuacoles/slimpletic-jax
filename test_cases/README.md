# Testing all of the things

This is a collection of test files which are being used to verify the behaviour of the new implementation.

- `harness` sets up the system
- `harness_1` tests coincidence of different components of the system
  1. That our lagrangians are equal
  2. That they are discretised correctly
  3. That `compute_pi_next` is correct
  4. That `compute_qi_values` is correct

This has identified that at the time of writing, the `compute_qi_values` function is not working correctly.

We now use `harness_2` implementing a hybrid model where we use the old `compute_qi_values` function but leave the rest of the computation the same. This has to use the `integrate_manual` entry point as the sympy code is not compatible with the JAX approach.

**Wait**: we are only getting one value out of the `compute_qi_values` function, this doesn't quite line up with what I thought.
We need to investigate this further.
