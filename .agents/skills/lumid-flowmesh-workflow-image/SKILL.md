---
name: lumid-flowmesh-workflow-image
description: Build and validate reproducible ML execution using Lumid Workflow, ephemeral FlowMesh workers, and custom container images published to Harbor. Use for Dockerfiles, image contents, Harbor publishing, FlowMesh YAML or DAG stages, runtime inputs, artifacts, and scheduled experiment execution. Do not use for Lumid Sandbox provisioning, persistence, package installation, or job management.
---

# Lumid Workflow and FlowMesh Images

Create a reproducible, platform-native path from a versioned container image to one or more FlowMesh workflow stages. Keep this path separate from Lumid Sandbox usage.

## Platform boundary

Use this mental model consistently:

| Component | Responsibility |
|---|---|
| Docker/OCI image | Repository code and pinned runtime dependencies |
| Harbor | Versioned image storage |
| Lumid Workflow | Execution definition and stage orchestration |
| FlowMesh | Scheduling and ephemeral worker containers |
| Sandbox | Separate persistent development environment; outside this skill |

Never describe a FlowMesh worker as a Sandbox or assume it inherits Sandbox state. A local-machine path, Sandbox home directory, virtual environment, or `/datasets` mount is unavailable to a worker unless the workflow explicitly stages or mounts it through a supported mechanism.

If a request concerns both systems, separate the work into a Workflow/FlowMesh track and a Sandbox track. Use the Sandbox-specific skill only for the latter.

## Required platform references

Before designing or changing a Workflow/FlowMesh implementation, read:

- `/mnt/e/School-Work-6-Y3S2/FYP/Lumid_guides/lumid-flowmesh-work-container-platform-guide.md` for the execution model, image guidance, FinData access, and storage constraints.
- `/mnt/e/School-Work-6-Y3S2/FYP/Lumid_guides/lumid_workflow_guide.md` for the current FlowMesh workflow dialect, graph stages, Python-node fields, metrics, and execution surface.

Read `/mnt/e/School-Work-6-Y3S2/FYP/Lumid_guides/lumid-flowmesh-ssh.md` only when the requested task is an interactive FlowMesh SSH session or when diagnosing an SSH-task image. Do not let SSH-task requirements dictate an ordinary non-interactive Python workflow.

Treat the platform guides as authoritative over examples in this skill if they change.

## Choose the execution shape

Prefer a non-interactive FlowMesh Python node or DAG for reproducible experiment execution. Use separate nodes when preparation, training, evaluation, and reporting need explicit dependencies or different resources.

Use a FlowMesh SSH task only for interactive worker debugging. For an SSH task, inherit the FlowMesh SSH base image and retain its required SSH server, entrypoint behavior, and `flowmesh` user. A generic Python or Ubuntu image is not an SSH-ready substitute.

## Image contract

When building an image for this repository:

1. Inspect `requirements.txt`, `requirements-container.txt`, imported system libraries, and the target worker's Python/CUDA/PyTorch compatibility. Do not blindly install both requirement files or copy the local `.venv`.
2. Build for `linux/amd64` unless current platform documentation specifies otherwise.
3. Pin the base image and dependency versions needed for the selected experiment stage. Use a versioned Harbor tag; avoid `latest` for a reproducibility claim.
4. Copy repository source, entry scripts, and required configuration into a stable working directory such as `/app`.
5. Decide explicitly how every dataset and checkpoint enters the container. Local presence alone is not sufficient. Either include a deliberately bounded frozen bundle in the image, stage it as a workflow input, or retrieve it from an approved persistent service.
6. Do not put credentials, PATs, Harbor secrets, private keys, `.env` files, or unrestricted research data into the image. Supply secrets through supported runtime environment mechanisms.
7. Use `.dockerignore` to exclude `.git`, local environments, caches, logs, and unrelated large experiment artifacts. If a demonstration intentionally embeds data or checkpoints, include only its documented immutable subset.
8. Check Harbor visibility before publishing. The current guide states that FlowMesh workers pull anonymously and therefore require a publicly pullable Harbor project. Do not publish private source or data without confirming an acceptable access mechanism.
9. Record the source commit, dependency lock or manifest, image tag, and preferably the resolved image digest.

## Workflow contract

For every FlowMesh node, define or verify the relevant:

- image reference and working directory;
- code/function or command wrapper that invokes the repository stage;
- connected inputs and artifact locations;
- runtime environment and secret names;
- network mode;
- CPU, memory, and GPU resources;
- timeout;
- declared metrics and returned outputs.

A workflow may use the same image for multiple stages while invoking different repository scripts. Keep platform wrappers thin: scientific logic should remain in `src/` and the existing versioned scripts.

FlowMesh containers are ephemeral. Return, upload, or otherwise persist checkpoints, predictions, metrics, logs, and reports before a task ends. Do not claim end-to-end reproducibility until both inputs and outputs have explicit lifecycle handling.

When FinData is needed, query it from the worker using the documented site-specific endpoint and runtime credentials. Do not rely on a Sandbox database attachment or Sandbox-mounted paths.

## Implementation sequence

1. Confirm the exact reproducibility target: a smoke test, frozen-model evaluation, one experiment stage, or fresh end-to-end training.
2. Inventory required code, packages, system libraries, prepared data, checkpoints, secrets, compute, and expected outputs.
3. Add narrowly scoped files under `docker/`, `workflows/`, or platform adapter scripts. Avoid refactoring scientific code merely to fit the platform.
4. Build the image and run a local smoke test that imports the project and exercises the intended entry point.
5. Push to Harbor only when the user has requested publishing and image contents are safe for the project's visibility.
6. Validate the FlowMesh YAML before consuming worker resources.
7. Run a minimal workflow before a costly experiment, then verify metrics, artifacts, logs, and failure propagation.
8. Record a concise reproducible command sequence and the platform identifiers needed to audit the run.

## Completion checks

Before calling the integration reproducible, verify that:

- a clean image build succeeds for the worker architecture;
- the container can import the project and run the selected stage without local `.venv` state;
- all required data and checkpoints are available inside the worker by an explicit mechanism;
- the Workflow validates and references the intended versioned image;
- no node assumes Sandbox or local-machine persistence;
- important outputs survive container termination;
- no secret or private artifact is baked into a publicly pulled image;
- the run command, source revision, image identity, inputs, and outputs are documented.
