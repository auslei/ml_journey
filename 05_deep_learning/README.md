# 05 Deep learning

Neural networks from one unit up to a tuned, regularised model. This part (05a) covers the foundations: how a network is built, how it learns, and what stops it learning. Convolutional and recurrent networks come in part 05b. Keras 3 on the PyTorch backend is the teaching API. Notebook 05 uses `torch` directly to read gradients off the layers.

## Concept map

```
unit -> layer -> network         loss: MSE, cross-entropy
        |                              |
        v                              v
   backprop (chain rule) ------> gradient per weight
        |                              |
        |            optimiser: SGD, momentum, RMSprop, Adam
        |            learning rate schedule: decay, warmup, one-cycle
        |                              |
        +--- unstable gradients: init, activation, batch norm, clipping
        |
        +--- overfitting: L2, dropout, early stopping, callbacks
        |
        +--- hyperparameter search: random search, KerasTuner
```

## Reading order

1. `01_neural_net_basics.ipynb`: perceptron, XOR, an MLP in NumPy, then Keras on MNIST (classification) and California housing (regression). Width, and what unscaled inputs do.
2. `02_backprop_and_autodiff.ipynb`: backprop by hand on a 3-4-1 network, checked against numerical gradients, torch and Keras. The Keras functional API with a wide-and-deep model. Builds on `01_linear_algebra_calculus/04_autodiff.ipynb`.
3. `03_optimisers.ipynb`: SGD, momentum, RMSprop and Adam on a curved 2D valley, then on MNIST. Learning rate sweeps and divergence.
4. `04_learning_rate_schedules.ipynb`: step, exponential, cosine, warmup and one-cycle, checked against Keras `CosineDecay`. A peak rate that only survives with warmup.
5. `05_unstable_gradients.ipynb`: gradient norms per layer for sigmoid, ReLU, Glorot and He. Batch norm, activation choice, exploding gradients and `clipnorm`.
6. `06_regularisation_and_callbacks.ipynb`: L2, dropout, early stopping written by hand and checked against Keras, then `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` and `CSVLogger`. TensorBoard is the dashboard version of the CSV log.
7. `07_hyperparameter_tuning.ipynb`: a seeded random search by hand, then KerasTuner, then the winner's curse.
8. `intuition_test.ipynb`: predict first, then run.

Data comes from `common/datasets.py`: MNIST subsets (300 to 5,000 images) and California housing (6,000 rows). Both are cached in `data/` on first use. Helpers shared across notebooks live in `common/`: `setup_keras`, `set_seed` and `weight_gradient_norms` in `common/dl.py`, and `plot_history` in `common/plotting.py`. The notebooks pick up a GPU on their own. The whole folder runs in about 3 minutes through `pytest 05_deep_learning`, mostly in Keras training calls.

## What you should be able to explain at the end

- Why a single unit cannot solve XOR and a hidden layer can, and what the activation function is for.
- Which loss goes with which output layer, and why inputs need scaling.
- How backprop applies the chain rule from the output back, why it costs about one forward pass, and how to check a gradient numerically.
- What the Keras functional API allows that `Sequential` does not.
- What momentum, RMSprop and Adam each add to plain gradient descent, and why the learning rate is usually the knob that matters.
- Why a learning rate that is stable late in training can be fatal at the start, and what warmup and decay do about it.
- Why gradients vanish through sigmoid layers, what He and Glorot initialisation keep constant, and what batch norm and clipping each fix.
- How L2 shrinks weights, why dropout is switched off at test time, and how early stopping picks its epoch.
- Why random search beats a grid, why learning rates are sampled on a log scale, and why the best trial's score is optimistic.
