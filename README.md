# RL-multiple-human-feedback

Source code for "RL with feedback from multiple humans with diverse skills"

A link to our paper can be found on [arXiv](https://arxiv.org/abs/2111.08596).

# Installation
Clone this repository
```
git clone https://github.com/jmerskine1/RL-multiple-human-feedback.git
```
Enter the repository
```
cd RL-multiple-human-feedback
```
Change to the branch we need
```
git checkout dev_2
```
Make a new python environment (using conda)
```
conda create -n PACMAN_exp python=3.11 -y
```
Install requirements
```
conda activate PACMAN_exp
cd human_interface
pip install -r requirements.txt
```
Run the experiment
```
python flask_app.py
```
