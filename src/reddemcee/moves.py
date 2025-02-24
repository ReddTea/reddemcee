# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np
from copy import deepcopy

from .state import State

class PTMove(object):
    def __init__(self):
        self.smd_hist = False
        self.D_ = np.array([1])
        self.inverted = False
        self.freeze = False
        
    def adapt(self, state, dSs, copy=True):
        dc = deepcopy if copy else lambda x: x
        
        ntemps, nwalkers, ndim = state.shape
        betas = dc(state.betas)

        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])
        dbetas = betas - state.betas

        state.betas = betas
        for t in range(ntemps-1, 0, -1):
            state[t].log_prob += state[t].log_like * dbetas[t]

        return state


    def update(self, state, s1, s2):
        pass


    def swap(self, state):
        betas = state.betas
        ntemps, nwalkers, ndim = state.shape
        n_swap_accept = np.zeros(ntemps-1)
        smd = np.zeros(ntemps-1) if self.smd_hist else None
        dbetas = -np.diff(betas)

        for t in self.beta_range(ntemps):
            state_hot = state[t]
            state_cold = state[t-1]

            ll1 = state_hot.log_like  # hot
            ll2 = state_cold.log_like  # cold

            dbeta = dbetas[t-1]  # replace

            raccept = np.log(np.random.uniform(size=nwalkers))
            paccept = dbeta * (ll1 - ll2)

            asel = paccept > raccept
            n_swap_accept[t-1] = np.sum(asel)

            AA = state_hot.coords[asel]
            BB = state_cold.coords[asel]
            if self.smd_hist:
                if asel.any():
                    a_min_b = (AA-BB)/self.D_
                    sup = np.sqrt(np.sum(np.square(a_min_b), axis=1))
                    zeros = np.zeros(int(nwalkers-n_swap_accept[t-1]))
                    smd[t-1] = np.mean(np.append(sup, zeros))
                else:
                    smd[t-1] = 0.

            state_hot.coords[asel], state_cold.coords[asel] = BB, AA
            state_hot.log_prob[asel], state_cold.log_prob[asel] = state_cold.log_prob[asel] - dbeta*ll2[asel], state_hot.log_prob[asel] + dbeta*ll1[asel]
            state_hot.log_like[asel], state_cold.log_like[asel] = ll2[asel], ll1[asel]

        return state, n_swap_accept/nwalkers, smd


    def beta_range(self, ntemps):
        self.inverted = not self.inverted
        return range(1, ntemps) if self.inverted else range(ntemps-1, 0, -1)



class Move(object):
    def tune(self, state, accepted):
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        Args:
            coords: The original ensemble coordinates.
            log_probs: The original log probabilities of the walkers.
            blobs: The original blobs.
            new_coords: The proposed coordinates.
            new_log_probs: The proposed log probabilities.
            new_blobs: The proposed blobs.
            accepted: A vector of booleans indicating which walkers were
                accepted.
            subset (Optional): A boolean mask indicating which walkers were
                included in the subset. This can be used, for example, when
                updating only the primary ensemble in a :class:`RedBlueMove`.

        """
        if subset is None:
            subset = np.ones(len(old_state.coords), dtype=bool)

        # integer indexing over boolean masks for better performance?
        subset_indices = np.where(subset)[0]
        accepted_subset = accepted[subset]
        m1 = subset_indices[accepted_subset]
        m2 = np.where(accepted_subset)[0]
        '''
        m1 = subset & accepted
        m2 = accepted[subset]
        '''
        # Update the old_state with the accepted proposals
        old_state.coords[m1] = new_state.coords[m2]
        old_state.log_prob[m1] = new_state.log_prob[m2]
        old_state.log_like[m1] = new_state.log_like[m2]


        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_prob, "
                    "you also need to provide the current list of "
                    "blobs at that position."
                )
            old_state.blobs[m1] = new_state.blobs[m2]

        return old_state


class RedBlueMove(Move):
    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False
    ):
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError(
            "The proposal must be implemented by " "subclasses"
        )

    def propose(self, model, state):
        """Generate a proposal for a move and compute the acceptance based on the proposed position.

        Args:
            model (Model): The model containing the log probability function and random number generator.
            state (State): The current state of the walkers, including their coordinates and log probabilities.

        Returns:
            tuple: A tuple containing:
                - state (State): The updated state of the walkers after the proposal.
                - accepted (np.ndarray): A boolean array indicating which proposals were accepted.
        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)

        # Get the two halves of the ensemble.
        #sets = [state.coords[inds == j] for j in range(self.nsplits)]
        for split in range(self.nsplits):
            S1 = inds == split

            s = state.coords[S1]
            c = state.coords[~S1]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_log_likes, new_blobs = model.compute_log_prob_fn(q, state.beta)
            #print(f'new_log_probs = {new_log_probs.shape}')

            # Loop over the walkers and update them accordingly.
            # Why this is not vectorised??????
            '''
            for i, (j, f, nlp) in enumerate(
                zip(all_inds[S1], factors, new_log_probs)
            ):
                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True
            '''
            lnpdiff = factors + new_log_probs - state.log_prob[S1]
            u = np.log(model.random.rand(len(lnpdiff)))
            accept = lnpdiff > u
            accepted[S1] = accept

            #print(f'subset = {S1.shape}; accepted = {accepted.shape}')
            #print(f'subset = {S1}; accepted = {accepted}')
            #print(f'q = {q.shape}; new_log_probs = {new_log_probs.shape}')
            new_state = State(q, log_prob=new_log_probs, log_like=new_log_likes, beta=state.beta, blobs=new_blobs)
            self.update(state, new_state, accepted, S1)

        return state, accepted


class StretchMove(RedBlueMove):

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        #c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a
        factors = (ndim - 1.0) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))
        return c[rint] - (c[rint] - s) * zz[:, None], factors

