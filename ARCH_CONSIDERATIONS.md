# Architecture Considerations

There are the following use cases to consider for this code:

The overal parameters of the system are:

`(r, dt, L, K, q0, pi0, t0, iterations)`

These fit into 3 different categories:

- If `r` changes, there is nothing we can do as the size of arrays changes and this hits JAX's equal array size requirement.
- If `dt` changes, `L_d, K_d` will take different values, and the physical effect of `iterations` will change, but sizes
  will remain the same.
- If `L, K` change then the behaviour of the system will change but not the solver's method, array sizes will remain
  fixed.
  - I am uncertain if we can avoid recomputing everything however
  - Note that the parameters we seek are **inside** these functions... ahh fuck
  - So almost we need to pass the embedding through the entire system to then be realised into their action only at each
    point of computation.
  - Reasons:
    - We need to have the values not in a static so that we can differentiate wrt to them.
    - We currently have the live values within the lambda stored in the static and this is not working
    - Hence, we need to pass them along
      - I probably want and need to register things as Pytrees but for now we can ignore shit

1. Method Parameters: `(r,)`
   - There is nothing we can do if these change as the
   