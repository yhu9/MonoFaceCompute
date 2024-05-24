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
conda env update -n $ENV_NAME --file ./environment.yaml 

# Install MediaPipe (we have to do it separately to avoid conflits with protobuf's version from face-detection-tflite)
pip install mediapipe==0.10.11
# Install ffmpeg separately for similar reasons
conda install ffmpeg~=4.3 -y

pip install -e ./submodules/INFERNO

#################### Omnidata ####################
# conda install -c conda-forge aria2
# (cd submodules/omnidata && pip install 'omnidata-tools')

#################### DSINE ####################
pip install geffnet