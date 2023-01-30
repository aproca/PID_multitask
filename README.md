# *Synergistic information supports modality integration and flexible learning in neural networks solving multiple tasks* 

by Alexandra M. Proca, Fernando E. Rosas, Andrea I. Luppi, Daniel Bor, Matthew Crosby*, Pedro A.M. Mediano*<br/>
(* – joint senior authorship)

Code and data for https://doi.org/10.48550/arXiv.2210.02996 .


### Code

To train, test, and compute PID measures for experiments in the paper, run this command:
```train
python3 logic_script.py --config <config to run>
python3 animalai_script.py --config <config to run>
python3 neurogym_script.py --config <config to run>
```

### Data
Trained models, stored activations, and computed PID measures used in the paper can be found in trained_models, activations, and PID folders, respectively.


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
