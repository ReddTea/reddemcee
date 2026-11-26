# API Reference

For a complete sampling workflow, see [Evidence with MBAR](user-guide/evidence.md).

## Documentation for `PTSampler`

::: reddemcee.sampler.PTSampler
    handler: python
    options:
        show_root_heading: false
        show_source: true
        members:
            - __init__
            - sample
            - run_mcmc
            - reset
            - get_evidence_ti
            - get_evidence_ss
            - get_evidence_hybrid
            - get_evidence_mbar
            - get_autocorr_time
            - get_betas
            - get_chain
            - get_blobs
            - get_log_prob
            - get_log_like
            - get_last_sample
            - get_value

## Evidence utilities

These functions accept saved beta and log-likelihood arrays. `get_mbar` is also
available as `from reddemcee import get_mbar`.

::: reddemcee.evidence
    handler: python
    options:
        show_root_heading: false
        show_source: true
        members:
            - get_mbar
            - get_bar
            - get_ti_corrected
