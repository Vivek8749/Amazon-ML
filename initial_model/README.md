# Initial model (CUDA)

The initial model requires an NVIDIA GPU and a working CUDA runtime. Install
the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

Run the standalone pipeline from `initial_model/`:

```bash
python src/run_pipeline.py --mode full --sample-size 50000
```

CUDA is checked at startup. XGBoost trains and predicts with `device="cuda"`,
and TF-IDF sparse similarity is evaluated with CuPy/CuPyX on the GPU. Pandas,
text normalisation, and RapidFuzz feature construction remain CPU host-side
because those libraries do not provide CUDA implementations.

Training data loading is cached under `../.cache/`. The first run scans the
large S2/S3 TSV files, samples the required pool before preprocessing it, and
saves that pool as Parquet. Repeated notebook runs with the same sample size
reuse the cache and avoid rescanning and re-normalising the pool.
