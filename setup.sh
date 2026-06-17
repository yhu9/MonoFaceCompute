ENV_NAME=MonoFaceCompute

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
pip install numpy==1.23
pip install -e ./submodules/INFERNO

#################### Omnidata ####################
# mamba install -c conda-forge aria2
# (cd submodules/omnidata && pip install 'omnidata-tools')

#################### DSINE ####################
pip install geffnet==1.0.2
