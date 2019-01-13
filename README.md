https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip

# unity-ml-crawler
This repository contains an implementation of reinforcement learning based on:
	* Proximal Policy Optimization with a Critic Network as a baseline and with a Generalized Advantage Estimation
The agent being training is a creature with 4 arms and 4 forearms. It has a 20 double-jointed arms. The agent rewards are
	* +0.03 times body velocity in the goal direction.
	* +0.01 times body direction alignment with goal direction. 
This environment is simliar to the [crawler of Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).<br/>
The action space is continuous [-1.0, +1.0] and consists of 20 values, corresponding to target rotations for joints.<br/>
`The environment is considered as solved if the average score of the 20 agents is +30 for 100 consecutive episodes.`<br/>
A video of a trained agent can be found by clicking on the image here below <br/>
* TO UPDATE --> PPO: [![Video](https://img.youtube.com/vi/E0uoV_c21w8/0.jpg)](https://www.youtube.com/watch?v=E0uoV_c21w8)
## Content of this repository
* folder __agents__: contains the implementation of
	* a Gaussian Actor Critic network for the PPO
	* an implementation of a Proximal Policy Optimization
* folder __weights__: 
	* weights of the Gaussian Actor Critic Network that solved this environment with PPO
* Notebooks
	* jupyter notebook __Continuous_Control-PPO-LeakyReLU.ipynb__: run this notebook to train the agents using PPO
## Requirements
To run the codes, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	* __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install .
```
* Fix an issue of pytorch 0.4.1 to allow backpropagate the torch.distribution.normal function up to its standard deviation parameter
    * change the line 69 of Anaconda3\envs\drlnd\Lib\site-packages\torch\distributions\utils.py
```python
# old line
# tensor_idxs = [i for i in range(len(values)) if values[i].__class__.__name__ == 'Tensor']
# new line
tensor_idxs = [i for i in range(len(values)) if isinstance(values[i], torch.Tensor)]
``` 
* Create an IPython kernel for the `drlnd` environment
```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	* [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	* [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

* Start jupyter notebook from the root of this python codes
```bash
jupyter notebook
```
* Once started, change the kernel through the menu `Kernel`>`Change kernel`>`drlnd`
* If necessary, inside the ipynb files, change the path to the unity environment appropriately

