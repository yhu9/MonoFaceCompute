ENV_NAME=MonoFaceCompute

# pull submodules
./pull_submodules.sh

# create conda environment
echo "Creating conda environment"
mamba create -n $ENV_NAME python=3.8
conda activate $ENV_NAME

# Environment cuda libraries
mamba install -n $ENV_NAME -c "nvidia/label/cuda-11.7.0" cuda-toolkit -y
mamba install -n $ENV_NAME -c conda-forge gxx_linux-64=11 gcc_linux-64=11 -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export CUDA_HOME=$CONDA_PREFIX
# which nvcc && nvcc --version

mamba env update -n $ENV_NAME -f environment.yaml

# PyTorch + Pytorch3d
mamba install -n $ENV_NAME pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
mamba install -n $ENV_NAME iopath -c iopath -y
# Pin torch/pytorch-cuda and add the pytorch/nvidia channels so the solver can't downgrade torch
mamba install -n $ENV_NAME pytorch3d pytorch=1.13.0 pytorch-cuda=11.7 -c pytorch3d -c pytorch -c nvidia -c conda-forge -y
# Install ffmpeg separately for similar reasons
mamba install ffmpeg~=4.3 -y
# Downgrade GCC
mamba install gcc=12.1.0 -c conda-forge -y

pip install mediapipe==0.10.9 protobuf==3.20.3
pip install -e ./submodules/INFERNO

#################### Omnidata ####################
# mamba install -c conda-forge aria2
# (cd submodules/omnidata && pip install 'omnidata-tools')

#################### DSINE ####################
pip install geffnet==1.0.2

#################### Final version pins ####################
# These must run LAST: earlier pip installs (e.g. INFERNO and its deps) pull in
# unpinned versions that override the project's pins, so we re-assert them here.

# Later pip installs upgrade numpy to 1.24.x (violates tensorflow's numpy<1.24);
# re-pin to 1.23 to match the rest of the project.
pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.23.0"

# conda pulls h5py from conda-forge but hdf5 from the defaults channel, whose
# build lacks the H5Pget_fapl_direct symbol -> "undefined symbol" ImportError.
# Reinstall h5py from the PyPI wheel, which bundles its own HDF5 and avoids the
# conda libhdf5 entirely.
pip install --no-cache-dir --force-reinstall --no-deps "h5py==3.7.0"

./download_all_assets.sh
