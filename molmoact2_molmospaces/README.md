# molmoact2_molmospaces — 🗓️ planned

Interactive playground for **MolmoAct2** driving the **MolmoSpaces** MuJoCo sim.

This is the "best of both" MolmoAct2 path: keep the live MuJoCo browser viewer (sim is
CPU, runs anywhere) while serving MolmoAct2 from a GPU pod. It needs a small adapter
because the schemas line up but the wire protocols differ:

- MolmoAct2 ships a FastAPI **`/act`** server (`json_numpy` HTTP) for the DROID Franka
  (external + wrist cameras, 8-D state) — the *same* embodiment MolmoSpaces' DROID sim uses.
- MolmoSpaces' learned policies are websocket/msgpack clients (openpi protocol).

So the work is one `InferencePolicy` subclass (the MolmoSpaces eval "external repo"
pattern) that HTTP-POSTs each observation to MolmoAct2's `/act` endpoint and unpacks the
returned action chunk — then reuse the same interactive web wrapper as
`molmobot_molmospaces/`.

Reference repos (gitignored clones): `../molmo/molmoact2/examples/droid/host_server_droid.py`,
`../molmo/molmospaces/molmo_spaces/evaluation/README.md` ("Implementing Eval in an External Repo").

Not started yet — `molmobot_molmospaces/` first, then this reuses its web/sim glue from
[`molmo_shared/`](../molmo_shared/).
