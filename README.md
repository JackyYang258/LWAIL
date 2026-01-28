# LWAIL: Latent Wasserstein Adversarial Imitation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ArXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]([INSERT_ARXIV_LINK_HERE])

This is the official PyTorch implementation of the paper **Latent Wasserstein Adversarial Imitation Learning (LWAIL)**.

[**Read the Paper**]([INSERT_ARXIV_LINK_HERE])

## 📝 Abstract
Imitation Learning (IL) enables agents to mimic expert behavior by learning from demonstrations. However, traditional IL methods require large amounts of medium-to-high-quality demonstrations as well as actions of expert demonstrations, both of which are often unavailable. To reduce this need, we propose Latent Wasserstein Adversarial Imitation Learning (LWAIL), a novel adversarial imitation learning framework that focuses on state-only distribution matching. It benefits from the Wasserstein distance computed in a dynamics-aware latent space. This dynamics-aware latent space differs from prior work and is obtained via a pre-training stage, where we train the Intention Conditioned Value Function (ICVF) to capture a dynamics-aware structure of the state space using a small set of randomly generated state-only data. We show that this enhances the policy’s understanding of state transitions, enabling the learning process to use only one
or a few state-only expert episodes to achieve expert-level performance. Through experiments on multiple MuJoCo environments, we demonstrate that our method outperforms prior Wasserstein-based IL methods and prior adversarial IL methods, achieving better results across various tasks.

---

## ⚙️ Environment Settings

To run the experiments, please set up the environment using the following commands.

### Prerequisites
* Python = 3.8
* PyTorch 2.4.1 with CUDA 12.4
* [D4RL]([https://github.com/Farama-Foundation/D4RL])
* [dm_control]([https://github.com/google-deepmind/dm_control])(optional, not required for main experiments)


### Installation
We recommend using Anaconda to manage the environment:

For other Python packages, install the required dependencies via:
```bash
pip install -r requirements.txt
```

Note: Other versions of Python and PyTorch may also work. The most important requirement is that D4RL and its dependencies are installed successfully.

## 📂 Models and Datasets

### Pre-trained Models

We provide our basic ICVF (Intent-Conditioned Value Function) model in the icvf_model/ directory, which can be loaded directly for evaluation or fine-tuning.

For other model variants or to access the training code for the base model, please refer to the external codebase:

ICVF-PyTorch Repository(todo)

### Datasets

Due to GitHub storage capacity limitations, this repository includes a standard one-trajectory dataset as a sample.

To generate the full datasets or acquire more trajectories:

Navigate to the datasets/ folder.

## 🚀 How to Run Experiments
We provide shell scripts to streamline the training and evaluation process.

### Single Experiment
To run a specific experiment configuration:


```Bash
bash run.bash
```

### Reproduce Main Results
To perform a quick reimplementation of the main experiments presented in the paper, execute the following script. This will run the training loop across the defined environments:

```Bash
bash all_run.bash
```

## 📍 Citation
If you find this code or paper useful for your research, please cite:

代码段
@article{YourName202X,
  title={Latent Wasserstein Adversarial Imitation Learning},
  author={Author One and Author Two and Author Three},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={202X}
}

## 📧 Contact
For any questions, please feel free to open an issue in this repository(prefered) or reach out to Siqi Yang at siqiyang@illinois.edu.