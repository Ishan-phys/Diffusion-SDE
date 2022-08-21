# Diffusion SDE - A score-based generative modelling with SDEs package

Synthesize new images using the score-based generative models.

# Installation

Currently, `diffusion_sde` supports release of Python 3.7 onwards.

To install the current release:

```shell
$ pip install -U diffusion_sde
```

# Getting Started 

Start by instantiating a dataset class with a path where the custom dataset is located

```python
from diffusion_sde import datasets

# Specify the path of the custom dataset in the dataset class
ds = datasets(path_to_dataset)
```

Then, instantiate the `diffSDE` class to train the model and generate samples and pass the dataset using `.set_loaders()` method

```python
from diffusion_sde import diffSDE

# Instantiate the diffSDE class
cls_diff = diffSDE()

# Set the dataloaders by passing the dataset instantiation as above
cls_diff.set_loaders(dataset=ds)
```

Begin the model training using the `.train()` method and select the desired number of epochs for training.

```python
# Train the model
cls_diff.train(n_iters)
```

Generate the samples from the trained model with the `.generate_samples()` method and specify the desired number of steps for the sampler. We suggest setting the value of `n_steps` in the range of $\sim1500$-$2000$ steps to produce high-quality samples

```python
# Generate samples from the trained model
cls_diff.generate_samples(n_steps)
```
