# Disentangled (Un)Controllable Features

This is the PyTorch repository for the paper : 

***Disentangled (un)Controllable Features***

By Jacob E. Kooi, Mark Hoogendoorn and Vincent Francois-Lavet 

(Link to Arxiv: http://arxiv.org/abs/2211.00086)


![Four Mazes](Github.png)

**Citation**

If you are using this code or are referrring to our work, please cite the following bibtex: 

```
@misc{https://doi.org/10.48550/arxiv.2211.00086,
  doi = {10.48550/ARXIV.2211.00086},
  url = {https://arxiv.org/abs/2211.00086},
  author = {Kooi, Jacob E. and Hoogendoorn, Mark and François-Lavet, Vincent},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Disentangled (Un)Controllable Features},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

**Requirements**

Python 3.8.10 was used, and the packages needed to run the code are:

```bash
pip3 install pytorch    # Pytorch 1.12
pip install matplotlib  # Matplotlib 3.6.1
pip install argparse    # Argparse 1.4.0
```

**Four Mazes environment**

To train an encoder and forward predictor for the Four Mazes environment, run:
```bash
python3 main_fourmaze.py --run_description test_fourmaze --iterations 50000
```


**Catcher environment**

To train an encoder and forward predictor for the Catcher environment, run:
```bash
python3 main_catcher.py --run_description test_catcher_1_state --iterations 200000 --agent_dim 1 --entropy_scaler 5
```
or with two agent dimensions:
```bash
python3 main_catcher.py --run_description test_catcher_2_states --iterations 200000 --agent_dim 2 --entropy_scaler 5
```
or with two agent dimensions and adversarial loss:
```bash
python3 main_catcher.py --run_description test_catcher_2_states_adversarial --iterations 200000 --agent_dim 2 --entropy_scaler 5 --adversarial True
```


**Procedural Generated Maze environment**

To train an encoder and forward predictor for the Random Maze environment, run:

```bash
python3 main_multimaze.py --run_description test_multimaze --iterations 250000 --entropy_scaler 8
```

**Procedural Generated Maze environment + Reinforcement Learning**

You can train a representation and apply reinforcement learning on it afterwards. You can also use the pretrained encoder and forward predictor saved in the directory *saved_models*. The training setting can be specified with the --mode argument. For instance:
```bash
python3 main_multimaze_modes.py --run_description test_DDQN_Only --iterations 500500 --mode dqn_only
```
trains a representation end-to-end using only a reinforcement learning agent, and 
```bash
python3 main_multimaze_modes.py --run_description test_saved_models --iterations 500500 --mode pretrain_saved_model
```
trains DDQN on a representation using the saved encoder and forward predictor seen in the figure below:

![Saved Representation](saved_representation.png)

**Contact**

For any questions, comments or general contact feel free to send an email to *j.e.kooi@vu.nl*!



