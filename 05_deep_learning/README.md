# 05 Deep learning

Neural networks from one unit up to convolutional, recurrent and generative models. Notebooks 01 to 07 (05a) cover the foundations: how a network is built, how it learns, and what stops it learning. Notebooks 08 to 12 (05b) cover the architectures: PyTorch and its training loop, CNNs, transfer learning, RNNs and a VAE. Keras 3 on the PyTorch backend is the teaching API for 01 to 07 and 09. Native PyTorch is used where the loop is the lesson (08, 10, 11, 12). TensorFlow-only material goes in an appendix, not yet written.

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

PyTorch loop (08) ---> CNN: shared filters, pooling (09) ---> transfer learning (10)
        |
        +--- sequences: RNN, vanishing gradient through time, GRU, LSTM (11)
        +--- generative: autoencoder, VAE, reparameterisation, KL (12)
```

## Reading order

1. `01_neural_net_basics.ipynb`: perceptron, XOR, an MLP in NumPy, then Keras on MNIST (classification) and California housing (regression). Width, and what unscaled inputs do.
2. `02_backprop_and_autodiff.ipynb`: backprop by hand on a 3-4-1 network, checked against numerical gradients, torch and Keras. The Keras functional API with a wide-and-deep model. Builds on `01_linear_algebra_calculus/04_autodiff.ipynb`.
3. `03_optimisers.ipynb`: SGD, momentum, RMSprop and Adam on a curved 2D valley, then on MNIST. Learning rate sweeps and divergence.
4. `04_learning_rate_schedules.ipynb`: step, exponential, cosine, warmup and one-cycle, checked against Keras `CosineDecay`. A peak rate that only survives with warmup.
5. `05_unstable_gradients.ipynb`: gradient norms per layer for sigmoid, ReLU, Glorot and He. Batch norm, activation choice, exploding gradients and `clipnorm`.
6. `06_regularisation_and_callbacks.ipynb`: L2, dropout, early stopping written by hand and checked against Keras, then `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` and `CSVLogger`. TensorBoard is the dashboard version of the CSV log.
7. `07_hyperparameter_tuning.ipynb`: a seeded random search by hand, then KerasTuner, then the winner's curse.
8. `08_pytorch_basics.ipynb`: tensors, autograd, `nn.Module`, a hand-written training loop with device handling. The three usual mistakes: no `zero_grad`, wrong mode, wrong dtype.
9. `09_cnn_mnist.ipynb`: convolution and pooling in NumPy, a Keras CNN against the MLP (about 8 times fewer parameters), shifted digits and augmentation.
10. `10_transfer_learning.ipynb`: pretrained ResNet18 on cats against dogs from CIFAR-10. Frozen features, a trained head, fine-tuning and a from-scratch control. Weights download into `data/torch` on first run and the notebook falls back to random weights if the download fails.
11. `11_rnn_and_lstm.ipynb`: a vanilla RNN in NumPy and its gradient decay through time, then `RNN`, `GRU` and `LSTM` on a delayed-recall task. Reports honestly where the textbook ranking did not show.
12. `12_vae.ipynb`: autoencoder against VAE on MNIST with a 2D latent space, reparameterisation, the KL term, sampling, $\beta$ and posterior collapse.
13. `intuition_test.ipynb`: predict first, then run. Twelve questions covering notebooks 01 to 12.

Data comes from `common/datasets.py`: MNIST subsets (300 to 5,000 images), CIFAR-10 (1,500 cat and dog images) and California housing (6,000 rows). All are cached in `data/` on first use. Helpers shared across notebooks live in `common/`: `setup_keras`, `set_seed` and `weight_gradient_norms` in `common/dl.py`, `plot_history` in `common/plotting.py`, `make_memory_task` and `one_hot` in `common/sequences.py`, and `load_resnet18` in `common/pretrained.py`. The notebooks pick up a GPU on their own and run on CPU too. Notebooks 08 to 12 and the intuition test take about 100 seconds through `pytest` on a CUDA GPU and about 140 seconds on CPU. Numbers quoted in the prose were read off a GPU run. On CPU, single results move by a few points (for example the fine-tuning rate sweep in notebook 10 and the seed-level results in notebook 11), and the patterns hold.

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
- What a tensor, autograd and `nn.Module` each do, why the training loop has four steps in that order, and what happens if `zero_grad` or `eval()` is missing.
- Why a convolution has far fewer parameters than a dense layer, what pooling does, and why a CNN still fails on shifted digits unless it is trained on them.
- What a frozen layer is, why batch norm must be in `eval()` mode when frozen, why fine-tuning needs a small learning rate, and why the input normalisation belongs to the model.
- Why backprop through time multiplies the same matrix over and over, what the two ways to get it wrong are, and what a gate adds. Also why one seed on a sequence task proves little.
- What the reparameterisation trick makes differentiable, what the KL term pays for, the reconstruction against sampling trade-off that $\beta$ controls, and what posterior collapse looks like.
