# Moves
Setting moves is similar to emcee:

```python
import emcee

my_moves = [[(emcee.moves.DEMove(), 0.8),
             (emcee.moves.DESnookerMove(), 0.2)] for _ in range(ntemps)]
sampler = reddemcee.PTSampler(nwalkers, ndim_,
                                  loglike, logprior,
                                  ntemps=ntemps,
                                  moves=my_moves)

```
## Currently in construction!