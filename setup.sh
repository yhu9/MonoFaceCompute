ENV_NAME=MonoFaceCompute

echo "Creating conda environment"
conda create -n $ENV_NAME python=3.8 
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
if echo $CONDA_PREFIX | grep $ENV_NAME
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script."
    exit
fi

echo "Installing dependencies"
mamba env update -n $ENV_NAME --file ./environment.yaml 

# PyTorch
mamba install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch
# PyTorch3D
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
# Install MediaPipe (we have to do it separately to avoid conflits with protobuf's version from face-detection-tflite)
pip install mediapipe==0.10.11
# Install ffmpeg separately for similar reasons
conda install ffmpeg~=4.3 -y
# Downgrade GCC
conda install gcc=12.1.0 -c conda-forge -y

pip install -e ./submodules/INFERNO

#################### Omnidata ####################
# conda install -c conda-forge aria2
# (cd submodules/omnidata && pip install 'omnidata-tools')

#################### DSINE ####################
pip install geffnet
