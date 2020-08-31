# Deep Generative Cross-modal On-body Accelerometer Data Synthesis from Videos

## Requirements:

1. Python 3 required (for python build)
2. Install PyTorch and dependencies from http://pytorch.org
3. Install required modules (numpy scipy pandas sklearn matplotlib statistics pyyaml).

> ```pip3 install -r requirement.txt```

## File Structure
<pre>
root
├── code
│   ├── baseline
│   ├── checkpoints
│   ├── dataloader
│   └── ...
│
└── data
    ├── ECM
    ├── LabEating
    └── MD2K
</pre>


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/HAbitsLab/SensorSynthesis.git
cd SensorSynthesis
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### Train Model

### Load Pre-trained Model and Test



## Step by Step 
### Generate Optical Flow
In folder `preprocess`:
> 1. Stanford ECM dataset: `python3 run_pwcnet_ecm.py`
> 2. MD2K Sense2StopSync dataset: use `convert2mp4.sh`, then `python3 run_pwcnet_MD2K.py`

#### (progress: ECM process done; MD2K dataset to be uploaded to server)

### Feature Extraction
I3D Feature Extraction is used:
> In folder `utils/feature_extraction`, run `python3 run_i3d_ecm_gpu.py` and `python3 run_i3d_MD2K_gpu.py`.

#### (progress: ECM and MD2K (data currently uploaded) running)


### VAE-GAN
> 1. 



## Contact
[Shibo Zhang](http://users.eecs.northwestern.edu/~szh702/) (shibozhang2015@u.northwestern.edu)

## References
```
@inproceedings{dummmy,
}
```
