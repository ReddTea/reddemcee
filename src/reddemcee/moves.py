# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np
from emcee.moves import DEMove as _DEMove

from .state import State

class PTMove(object):
    def __init__(self):
        self.smd_hist = False
        self.tsw_hist = False
        self.D_ = np.array([1])
        self.inverted = False
        self.freeze = False
        
    def adapt(self, state, dSs, copy=True):
        # Rebuilding inverse gaps can shift a frozen ladder by one ULP.
        if not np.any(dSs):
            return state
        ntemps, nwalkers, ndim = state.shape
        previous_betas = state.betas
        # PTState.betas is already a fresh numeric array.
        betas = previous_betas.copy()

        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])
        dbetas = betas - previous_betas

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
        distances = np.zeros(nwalkers) if self.smd_hist else None
        dbetas = -np.diff(betas)

        for t in self.beta_range(ntemps):
            state_hot = state[t]
            state_cold = state[t-1]

            ll1 = state_hot.log_like  # hot
            ll2 = state_cold.log_like  # cold

            dbeta = dbetas[t-1]  # replace

            raccept = np.log(np.random.uniform(size=nwalkers))
            paccept = dbeta * (ll1 - ll2)

            asel = np.nonzero(paccept > raccept)[0]
            naccepted = len(asel)
            n_swap_accept[t-1] = naccepted
            if not naccepted:
                continue

            AA = state_hot.coords[asel]
            BB = state_cold.coords[asel]
            if self.smd_hist:
                a_min_b = (AA-BB)/self.D_
                distances[:naccepted] = np.sqrt(np.sum(np.square(a_min_b), axis=1))
                distances[naccepted:] = 0.
                # Preserve the previous padded reduction order exactly.
                smd[t-1] = np.mean(distances)

            state_hot.coords[asel], state_cold.coords[asel] = BB, AA
            state_hot.log_prob[asel], state_cold.log_prob[asel] = state_cold.log_prob[asel] - dbeta*ll2[asel], state_hot.log_prob[asel] + dbeta*ll1[asel]
            state_hot.log_like[asel], state_cold.log_like[asel] = ll2[asel], ll1[asel]
            if state_hot.blobs is not None:
                state_hot.blobs[asel], state_cold.blobs[asel] = (
                    state_cold.blobs[asel], state_hot.blobs[asel]
                )

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
            m1 = m2 = np.nonzero(accepted)[0]
        else:
            m2 = np.nonzero(accepted[subset])[0]
            m1 = np.nonzero(subset)[0][m2]
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
    min_complement = 1

    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False
    ):
        self.nsplits = int(nsplits)
        if self.nsplits != nsplits or self.nsplits < 2:
            raise ValueError("nsplits must be an integer greater than one")
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
        if nwalkers < self.nsplits:
            raise RuntimeError("A red-blue move requires at least nsplits walkers")
        smallest_complement = nwalkers - (nwalkers + self.nsplits - 1) // self.nsplits
        if smallest_complement < self.min_complement:
            raise RuntimeError(
                "This move requires at least {} complementary walkers".format(
                    self.min_complement
                )
            )
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
            subset_indices = np.nonzero(S1)[0]

            s = state.coords[subset_indices]
            c = state.coords[np.nonzero(~S1)[0]]

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
            lnpdiff = factors + new_log_probs - state.log_prob[subset_indices]
            u = np.log(model.random.rand(len(lnpdiff)))
            accept = lnpdiff > u
            accepted[subset_indices] = accept

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
        chosen = c[rint]
        return chosen - (chosen - s) * zz[:, None], factors


class DifferentialEvolutionMove(RedBlueMove):
    """Use emcee's differential evolution proposal in a tempered ensemble."""

    min_complement = 2

    def __init__(self, sigma=1.0e-5, gamma0=None, **kwargs):
        super().__init__(**kwargs)
        self._move = _DEMove(sigma=sigma, gamma0=gamma0)

    def setup(self, coords):
        self._move.setup(coords)

    def get_proposal(self, sample, complement, random):
        return self._move.get_proposal(sample, [complement], random)
