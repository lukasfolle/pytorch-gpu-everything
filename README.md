![Python application](https://github.com/lukasfolle/pytorch-gpu-everything/workflows/Python%20application/badge.svg)

# pytorch-gpu-everything
Contribution to pytorch hackathon at https://pytorch2020.devpost.com/

## Getting started

#### Conda installation

1. Create conda environment: 
`$ conda create --name team-gpu python=3.7 -y`
2. Activate environment: 
`$ conda activate team-gpu`

#### Pip installation

1. It is highly recommended to create a virtual environment: 
`$ virtualenv -p python3.7 venv`
`$ source venv/bin/activate`
2. Install necessary packages: 
`$ pip install -r requirements.txt`

#### Pipenv installation

1. Create a pipenv environment using Python 3.7:
`$ pipenv --python 3.7`
2. Install necessary packages:
`$ pipenv install -r requirements.txt`

> **_Note:_**  If you want to make use of GPUs on your system make sure to use `pytorch torchvision cudatoolkit=10.2`.
