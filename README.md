# Synergistic information supports modality integration and flexible learning in neural networks solving multiple tasks 

by Alexandra M. Proca, Fernando E. Rosas, Andrea I. Luppi, Daniel Bor, Matthew Crosby*, Pedro A.M. Mediano*<br/>
(* – joint senior authorship)

Code and data for https://doi.org/10.48550/arXiv.2210.02996 .

We study the partial information decomposition (PID; https://arxiv.org/abs/1004.2515) of small neural networks in several different settings-- supervised learning (XOR and COPY logic gates), reinforcement learning (AnimalAI), and neuroscience-inspired cognitive tasks (Neurogym). For each experiment, we train ten models, extract their activations during evaluation, and compute PID measures of the activations. We use two different redundancy functions, namely Immi (https://arxiv.org/abs/1411.2832) and Imin (https://arxiv.org/abs/1004.2515). In logic gate and AnimalAI experiments, we discretize activations by binning, while in Neurogym experiments, we compute continuous measures using a Gaussian copula. We also perform a lesioning experiment on logic gate networks by removing the maximal/minimal synergistic neuron in each layer iteratively and evaluating performance.


## Setup
To set up the conda environment, run:
```
conda env create -f environment.yml
conda activate environment
```
To run AnimalAI experiments, see Animal-AI 3 v2.2.3 for environment installation: https://github.com/mdcrosby/animal-ai (download the AAI repository and environment in the current folder for full functionality).<br/>
<b>Note:</b> v2.2.3 is an earlier version of the environment- it may or may not be compatable with newer versions.


## Code

Curriculums for AnimalAI and Neurogym experiments can be found in ```src/configs/AAI_curriculums.py``` and ```src/configs/NG_curriculums.py```, respectively. Hyperparameters for each experimental setting can be found in their respective parser file (```src/parsers/```) and modified. To train, test, and compute PID measures for experiments in the paper, run the following commands. <br/>

For logic gate experiments run:
```train
python3 logic_script.py --dataset=XOR --model_out_dir=trained_models/logic/XOR/ --activation_out_dir=activations/logic/XOR/ --PID_out_dir=PID/logic/XOR/ --log_out_dir=logs/logic/XOR/
python3 logic_script.py --dataset=COPY --model_out_dir=trained_models/logic/COPY/ --activation_out_dir=activations/logic/COPY/ --PID_out_dir=PID/logic/COPY/ --log_out_dir=logs/logic/COPY/
```

For AnimalAI experiments run:
```train
python3 animalai_script.py --curriculum=<curriculum name to run>
```

For Neurogym experiments run:
```train
python3 neurogym_script.py --curriculum=<curriculum name to run> --interleaved=<boolean whether interleaved>
```

## Data
Trained models, stored activations, and computed PID measures used in the paper can be found in ```trained_models```, ```activations```, and ```PID``` folders, respectively.

## Figures
Figures from the main body of the paper are plotted in ```figures.ipynb```


## Citation
Please cite our paper if you use this code in your research project.

```
@article{PIDmultitask2022,
  url = {https://arxiv.org/abs/2210.02996},
  author = {Proca, Alexandra M. and Rosas, Fernando E. and Luppi, Andrea I. and Bor, Daniel and Crosby, Matthew and Mediano, Pedro A. M.},
  title = {Synergistic information supports modality integration and flexible learning in neural networks solving multiple tasks},
  publisher = {arXiv},
  year = {2022}
}
```
