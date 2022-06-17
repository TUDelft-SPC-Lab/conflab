# Conflab dataset repository

This repository contains code that may be useful for use of the Conflab dataset.

We provide:

- Baselines used in the paper: 
    1. video pose estimation and person detection
    2. f-formation detection
    3. speaking status detection from pose and acceleration

- Preprocessing code used to process the raw data.
- Data loaders. We provide the data loaders (torch datasets) used to run our baselines.
- Analysis. Contains some of the code used to produce the visualizations in the paper.


# Setup



To use the code externally, add the parent folder to PYTHONPATH. For example:

```
git clone git@github.com:TUDelft-SPC-Lab/conflab.git
export PYTHONPATH=$PYTHONPATH:$PWD
```

Then you will be able to import conflab in your own projects. For example, to use our person data loader in your own project:

```
from conflab.data_loading.person import ConflabPersonDataset
```

## Path to the dataset

The dataloaders and preprocessing scripts need to know the path to the conflab dataset files. You can set it in constants.py:

```
conflab_path = '/path/to/my/conflab/copy'
```

## Baselines

The speaking status detection baseline is part of this repository (except for the MS-G3D pose-based method which is in a fork of the original author's repository):

- [Speaking status detection from pose](https://github.com/josedvq/MS-G3D)

The F-formations and person detection / keypoint estimation baselines are in separate repositories:

- [F-formation detection](https://github.com/steph-tan/FF_conflab)
- [Keypoint estimation](https://github.com/TUDelft-SPC-Lab/conflab-keypoint-detection)


All these baselines are also referenced as git submodules in this repository under `/baselines`. To download them, run the following after cloning:

```
git submodule init
git submodule update
```

or clone with `--recursive` option:

```
git clone --recursive git@github.com:TUDelft-SPC-Lab/conflab.git
```



