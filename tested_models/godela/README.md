# Godela - Agentic

Mesh-to-CAD by [Godela](https://godela.ai)'s agentic system, on the clean meshes. One run (`r1`); no noisy-mesh submission.
The method is proprietary; Godela can provide API access on request to reproduce or verify these results.

Every benchmark id has a row (3,000 per file). 30 rows are empty (no usable program) and count as failures.
Comments and docstrings were removed from the generated programs; the code is otherwise unchanged.

## Scoring

This repository's unmodified pipeline at commit 99e41a2, on `DeCoDELab/CADBench` revision 7cf5b3ec:

```
python scripts/run_eval_pipeline.py --model_dir tested_models/godela --variable-name result --num-workers 20
```

- Environment: conda Python 3.11.17, cadquery 2.7.0 (OCP 7.8.1.2), numpy 2.1.0, pymeshlab 2025.7.post1, trimesh 5.2.0
- Machine: AWS r7i.48xlarge, 192 vCPU, Intel Xeon Platinum 8488C, 1488 GiB; 20 workers, single-threaded BLAS; 86 min wall
