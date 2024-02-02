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
> You may encounter an issue where your connection to a local server is denied. If this is the case you will need to install and run redis-server

### Redis Server for Mac
```
brew install redis
```
then run the server with 
```
redis-server
```
Full installation instructions here: 
### Running on Ubuntu
For Ubuntu you need to install the redis server using
```
sudo apt install redis-server
```

### Running on Windows
Redis-server is not officially supported so windows installation is more involved.

Full installations instructions for each OS can be found here <href>https://redis.io/docs/install/install-redis/</href>
