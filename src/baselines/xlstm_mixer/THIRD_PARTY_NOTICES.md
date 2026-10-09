# xLSTM-Mixer third-party boundary

The Phase 6.9 adapter in this directory is independently authored. It uses
the following upstream projects as declared dependencies or behavioral
references; their licences are not replaced by the licence of this project.

- `mauricekraus/xlstm-mixer`, commit
  `730b0531aa9456e498765028f3c22ca3677de42e`, MIT licence. The project adapter
  follows the audited `FULL` forward-path behavior without vendoring the
  upstream training repository.
- `NX-AI/xlstm`, release `v1.0.3`, commit
  `1ff240242795062e56b4e39b43023cce61e8e88c`, AGPL-3.0 licence. This package
  supplies the sLSTM block stack at runtime and must be installed and
  distributed in accordance with its own terms.

The runtime admission manifest must record the resolved dependency version,
backend, environment, and applicable licence notice before training.

The dedicated local `.venv-xlstm-mixer` applies the explicitly modified
`scripts_v8/xlstm-1.0.3-lazy-cuda-init.patch` to the installed AGPL-3.0 package
(2026-10-09). It defers compiler-library discovery until the compiled loader
is called and removes the obsolete `include_paths(cuda=True)` call. Vanilla
sLSTM computation and parameters are unchanged. The original pinned source
is not claimed to be bit-for-bit unchanged; resolved source hashes must be
recorded, and the patch remains available with this repository. This local
runtime is not admission for a canonical Phase 6.9 training trajectory.
