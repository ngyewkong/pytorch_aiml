# PyTorch Ultimate 2024 Udemy Course Working Repo

## Env Setup

- conda env create -f pytorch_3_10.yaml

### if yaml file does not work (manual installation)

- conda create -n pytorch_3_10 python=3.10
- conda activate pytorch_3_10
- conda install pytorch torchvision torchaudio cpuonly -c pytorch
- conda install ipykernel
- conda install -c anaconda seaborn
- conda install scikit-learn
- conda install -c conda-forge detecto

### to export conda env dependency file

- conda env export > pytorch_3_10.yaml
