#!/bin/bash
# Run a python script inside the LUCiD container on a held S3DF GPU allocation, with the CUDA jax
# user-site env and this checkout (plus JAXTPC_PATH, if set) first on the python path.
#
#   HOLD_JOBID=<running slurm job> next/run_gpu_s3df.sh next/point_source_rays.py [args...]
#
# LUCID_IMAGE_PATH is the lucid .sif; LUCID_ENV_BASE the CUDA jax env (see
# docs/guides/production/deploy-s3df.md, "Running the JAX stack on a GPU").
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${HOLD_JOBID:?set HOLD_JOBID to a running GPU job to srun --overlap into}"
: "${LUCID_IMAGE_PATH:?set LUCID_IMAGE_PATH to the lucid .sif}"
LUCID_ENV_BASE=${LUCID_ENV_BASE:-/sdf/data/neutrino/$USER/python_envs/lucid}
export APPTAINERENV_PYTHONUSERBASE=$LUCID_ENV_BASE
export APPTAINERENV_PYTHONPATH=$REPO${JAXTPC_PATH:+:$JAXTPC_PATH}
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
exec srun --jobid="$HOLD_JOBID" --overlap --ntasks=1 apptainer exec --nv -B /sdf,/fs,/sdf/scratch \
  -B "$REPO":/opt/LUCiD "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 "$@"
