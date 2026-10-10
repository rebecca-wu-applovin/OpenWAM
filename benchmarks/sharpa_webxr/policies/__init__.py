"""Per-model policy servers on the policies/common.py contract (each runs in the training .venv, on a GPU).

A server owns its model's whole I/O (image layout, state encoding, normalization, sampler, action decode) and how
it is deployed upstream (execute_steps, past-observation rows), and publishes both through its "meta" endpoint.
The sim side (rollout.py) is the same for every model.
"""
