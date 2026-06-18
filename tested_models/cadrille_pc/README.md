Tested on Annie's blackwell workstation (Annie/cadrille)

1. Downloaded a split's stls into the data dir, like data/bench0_stls using:

```
python download_from_hf.py --split bench0 --out data/bench0_stls --format stl
```

2. Run normalization of meshes, e.g.:

```
python normalize_mesh_unitcube.py --input-dir data/bench0_stls --output-dir data/bench0_stls_norm
```

3. Run testing inside docker container:

```
sudo docker run --gpus all -it   --shm-size=16g   -v $(pwd):/workspace   -w /workspace   cadrille bash
```

```
python test.py --data-path ./data --split bench0_stls_norm --mode pc
```

4. Consolidate python output into json:

```
python collect_json.py --input-dir ./work_dirs/tmp_py --output-file results.json
```