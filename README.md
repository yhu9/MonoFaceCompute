Current setup steps in the original repo doesn't cover several details that needs to be addressed in order to run this repository. In order to address those issues this fork is updated with the correct steps necessary to get this running as of --------.

# MonoFaceCompute

This repository aims to facilitate preprocessing of monocular human face videos, covering a range of commonly used outputs from semantic segmentation to face tracking. The goal is to provide a convenient and coherent repository for research work.

Computations include: 
- Semantic Segmentation (https://github.com/zllrunning/face-parsing.PyTorch)
- Matting (https://github.com/ZHKKKe/MODNet)
- FAN landmarks (https://github.com/1adrianb/face-alignment)
- MediaPipe landmarks (https://github.com/google-ai-edge/mediapipe)
- Face tracking (FLAME)
    - DECA, EMOCA, FaceReconstruction: https://github.com/radekd91/inferno
    - SMIRK: https://github.com/georgeretsi/smirk
- Normals estimation
    - DSINE: https://github.com/baegwangbin/DSINE
    - omnidata: https://github.com/EPFL-VILAB/omnidata

Pull requests for other computations are welcome!

## Setup

The current setup step in the main branch doesn't currently work due to deprecation of older versions of numpy with tensorflow, pytorch, chumpy. The build process needs to be reworked so that the listed packages installed in the environment.yml file actually is able to be installed together. 

1. Pull the submodules: `./pull_submodules.sh`
2. Run the setup script: ./setup.sh to build a conda environment with all required dependencies.
3. Download pretrained models and other required files: `./download_all_assets.sh`
4. Configure your dataset according to the examples in [datasets](./datasets).

One dataset consists of one or multiple monocular videos. Several parameters can be tweaked, such as the strategy for cropping the videos, what face tracker to use, what dimensions the crops should be resized to or what steps of the preprocessing pipeline to run.

This was tested on Ubuntu 22.04 with a NVIDIA A5000 GPU.

## Usage

All computations are aggregated in a single entry point. Run the following command to process one dataset:
```bash
python process.py --datasets datasets/example.yaml
```

By default, the script will run the following steps:
- Video extraction using FFMPEG
- Face detection and cropping
- Matting
- Semantic segmentation
- Landmarks detection
- Tracking
- Tracking refinement through a landmarks-based optimization

## Dataset config

Supported fields of the dataset configuration files:

| Parameter | Help |
| --- | --- |
| base_dir | Base directory from which to retrieve the video(s). |
| output_dir | Where to save the processed data. |
| shape_sequence | Name of the sequence to use for estimating face shape. |
| crop_mode | fixed | constant | smooth (can be overriden per video) |
| crop_scale | Scaling factor for the detected face boxes for cropping. |
| resize | What size to resize the cropped image. |
| smooth_tracking | Apply a low-pass filter to the optimized pose and expression values. |
| tracker | What face tracker to use (DECA / EMOCA / FaceReconstruction / SMIRK). |
| shape_tracker | Optionally specify a different face tracker for recovering shape parameters (DECA / EMOCA / FaceReconstruction / SMIRK). |
| steps | What steps to launch (extract, crop, matte, segment, landmarks, track, optimize)
| sequences | Array of:<ul><li>source: input video file, relative to *base_dir* (e.g. "1.mp4")</li><li>crop_mode: fixed / constant / smooth</li><li>face_selection_strategy: strategy to use for selecting a detection when there are multiple (max_confidence / leftmost / rightmost) (only used if crop_mode=constant or crop_mode=smooth)</li><li>fixed_crop: [center_x, center_y, size] (only used if crop_mode=fixed)</li></ol>

## License

We refer to the individual submodules for their licensing information.  
MonoFaceCompute itself is provided under a Attribution-NonCommercial-ShareAlike 4.0 [license](./LICENSE).
