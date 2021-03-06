
# Regularization in intrinsically sparse networks

In 2017, Mocanu et al. pubslihed a remarkable neuroevolutionary scheme for training sparse neural nets (arxiv: 1707.04780, then appearing in Nature Communications **9** (19 June 2018), open source repository: https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks )

One starts with initializing sparse layers with random connectivity and then trains by repeating the following 2-step cycle a number of times: 
  * do some training, 
  * delete a fraction of connections with weights closest to zero, and recreate random new connections instead of the deleted ones.

The work was done for feedforward neural nets and for restricted Boltzmann machines. For the case of restricted Boltzmann machines the authors also demonstrated the ability of the system to learn advantageous network topology, forming higher density of connections in the active zone and lower density of connections in the non-meaningful margins.

---

In November 2018, Michael Klear implemented this scheme in PyTorch for feedforward neural nets and wrote a related blog post at https://towardsdatascience.com/the-sparse-future-of-deep-learning-bce05e8e094a

He demonstrated the ability of the system to learn the network topology contrasting the active zone and the margin for the case of feedforward neural nets.

---

I became interested in this implementation in February 2019, because I wanted to experiment with this neuroevolutionary scheme, and because PyTorch is my favorite machine learning platform at the moment.

When I looked more closely, I noticed the following strange effect: the results reported by Michael Klear for neural nets demonstrated **inverse pattern of network topology learning** compared to the results reported by Mocanu et al. for restricted Boltzmann machines. Namely, the density of connections in this case was **lower in the active zone and higher in the margin**.

I opened an issue in connection with this observation:

https://github.com/AlliedToasters/synapses/issues/1

and then I created this fork to investigate the situation experimentally.

This text is a write-up for this experimental investigation.

## Conjectures and high-level description of experimental findings

We will call the network topology learning demonstrated by Mocanu et al **positive learning**, and we will call the inverse pattern of the network topology learning emerging in the runs performed by Michael Klear **negative learning**. ("Negative" here does not a priori imply "bad", although as we shall see below, _in this series of experiments negative learning is usually associated with some overfitting/failure to generalize_.)

---

The main conjecture I made was that the **negative learning** effect was related to the absence of regularization in the original code. The logic I followed was that in the absence of regularization, when the weights pointing from the outlying areas are created, they tend to remain unchanged by training. At the same time, meaningful connections are changed by training, and occasionally become small and therefore get eliminated more frequently.

At the same time, if one were to add a sufficiently strong regularization encouraging smaller weights, then one would expect the connections which are not informative to the result to decrease on average more rapidly, than the connections which are informative.

The experiments I performed here and am describing below seem to confirm this conjecture. Namely, **when one adds sufficiently strong reglarization, negative learning is replaced by positive learning, and the stronger regularization is, the more pronounced is this effect**.

---

I do observe one more puzzling effect during those baseline runs which demonstrate overfitting/failure to generalize. There are two measures of quality involved here: the **loss function** for which one trains, and **accuracy**. Usually, trends in the loss function and in the accuracy go hand in hand, both in training and in test (validation): they tend to improve simultaneously. However, when the overfitting/failure to generalize is serious, the loss function in test (validation) stops improving and starts to get worse and worse, while the loss function in training keep converging. 

However, the test (validation) accuracy does not get worse and worse; in fact, it seems to tend to keep improving slowly despite deterioration in the test (validation) loss caused by overfitting (as if the system keeps getting gradually better in making the right predictions, despite becoming less confident in their correctness). Why this is so remains a mystery to me. (It turns out that this is a frequently observed phenomenon in various overfitting scenarios; I have not seen a good write-up on it yet.)

---

The remainder of this write-up includes the following sections:

  * Details of experiments with regularization
  * Baseline study
  * Future work


## Details of experiments with regularization

I mostly work with the first experiment conducted by Michael Klear (the one which involves `SparseNet` class), although at the end of this write-up I describe a bit of experimental work with the subsequent triplet of experiments there (they involve `EvolNet` class).

The identical runs give results in the same ballpark, but do not reproduce precisely despite the presence of `torch.manual_seed(0)` random seed setting in the original code (it might be that all that is needed is to move this random seed setting closer to the beginning of the code).

I studied effects of L2-regularization via the protocol recommended by PyTorch, namely adding `weight_decay` parameter to the optimizer, e.g. replacing

```python
optimizer = optim.SGD(sparse_net.parameters(), lr=lr, momentum=momentum)
```

with

```python
optimizer = optim.SGD(sparse_net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-3)
```

I started with (unrecorded) experiments with `weight_decay=1e-5` and `weight_decay=1e-4`, and the results were unremarkable: first there were signs of some moderate positive learning, which tended to give way to some moderate negative learning around the time when the signs of overfitting/failing to generalize started to appear. It was not obvious if there was much improvement compared to the baseline (it might be that the reason for needing higher `weight_decay` coefficients is that we have less weights than usual because of sparsity; I did succeed eventually with `weight_decay=1e-3` and `weight_decay=1e-2`).

At that moment I realized that better instrumentation is needed.

Because the effects are visually subtle in this experiment, and one is always asking oneself, "do I actually see this here (the preponderance of blue or yellow in the central area)?", I added the trend curves showing average connectivity values for squares with sides 2, 4, 6, .., 26, 28 around the center of the heat map, by modifying

```python
def show_MNIST_connections(model):
    vec = model.set1.connections[:, 0]
    vec = np.array(vec)
    _, counts = np.unique(vec, return_counts=True)
    sns.heatmap(counts.reshape(28, 28), cmap='viridis', xticklabels=[], yticklabels=[], square=True);
    plt.title('Connections per input pixel');
    plt.show();
```

to become

```python
def show_MNIST_connections(model):
    vec = model.set1.connections[:, 0]
    vec = np.array(vec)
    _, counts = np.unique(vec, return_counts=True)
    t = counts.reshape(28, 28)
    sns.heatmap(t, cmap='viridis', xticklabels=[], yticklabels=[], square=True);
    plt.title('Connections per input pixel');
    plt.show();
    v = [t[13-i:15+i,13-i:15+i].mean() for i in range(14)]
    plt.plot(v)
    plt.show()
```

The first couple of points on those trend curves are too jittery to matter much, but the rest give a good idea of how the connectivity changes from the center of the square to its borders.

I also changed the way the first experiment instrumented, in order to keep better track of train and test dynamics, replacing

```python
for epoch in range(1, epochs + 1):
    #In the paper, evolutions occur on each epoch
    show_MNIST_connections(sparse_net)
    if epoch != 1:
        sparse_net.evolve_connections()
    set_history = train(log_interval, sparse_net, device, train_loader, optimizer, epoch, set_history)
    #And smallest connections are removed during inference.
    sparse_net.zero_connections()
    set_history = test(sparse_net, device, test_loader, set_history)
    set_history.plot()
```

with

```python
for epoch in range(1, epochs + 1):
    #In the paper, evolutions occur on each epoch
    if epoch != 1:
        set_history.plot()
    show_MNIST_connections(sparse_net)
    if epoch != 1:
        print('Train set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
            set_history.train_loss[epoch-2], 100. * set_history.train_acc[epoch-2]))
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
            set_history.val_loss[epoch-2], 100. * set_history.val_acc[epoch-2]))
        sparse_net.evolve_connections()
        show_MNIST_connections(sparse_net)
    set_history = train(log_interval, sparse_net, device, train_loader, optimizer, epoch, set_history)
    #And smallest connections are removed during inference.
    sparse_net.zero_connections()
    set_history = test(sparse_net, device, test_loader, set_history)
    time.sleep(10) # YOU MIGHT WANT TO CHANGE THIS NUMBER OR REMOVE THIS LINE
```

---

Let's proceed to the actual experimental runs:

### weight_decay = 1e-3, 155 epochs

The first experimental notebook confirmes the conjecture that negative learning is caused by the lack of regularization, and that with proper regularization one obtains positive learning. The experiment

https://github.com/anhinga/synapses/blob/master/Experiment_1.ipynb

```
Train set: Average loss: 0.0795, Accuracy: 97.80%
Test set: Average loss: 0.0885, Accuracy: 97.30%
```

exhibits nice positive learning of network topology, while test (validation) performance is even money with

https://github.com/anhinga/synapses/blob/master/Baseline_1.ipynb

```
Train set: Average loss: 0.0428, Accuracy: 98.69%
Test set: Average loss: 0.0884, Accuracy: 97.21%
```

This level of regularization makes the model generalize nicely, compared to the baseline.

### weight_decay = 1e-3, 1000 epochs

Same here:

https://github.com/anhinga/synapses/blob/master/Experiment_1_complete.ipynb

```
Train set: Average loss: 0.0480, Accuracy: 98.75%
Test set: Average loss: 0.0720, Accuracy: 97.77%
```

ok positive learning of network topology, with test (validation) accuracy on par with the first experiment in

https://github.com/anhinga/synapses/blob/master/Baseline_complete.ipynb

```
Train set: Average loss: 0.0025, Accuracy: 99.96%
Test set: Average loss: 0.1262, Accuracy: 97.74%
```

but with loss function being much better (in terms of the loss function baseline overfits badly while showing a negative learning of network topology).

### weight_decay = 1e-2, 1000 epochs

A similar run with 1e-2 regularization, which learns the network topology even better. It demonstrates a much strong positive learning of the network topology compared to 1e-3, but underfitting is well pronounced. It converges way too slowly, and it's not clear whether it would eventually reach the good numbers, if one lets it train long enough:

https://github.com/anhinga/synapses/blob/master/Experiment_2.ipynb

```
Train set: Average loss: 0.1583, Accuracy: 96.02%
Test set: Average loss: 0.1512, Accuracy: 96.22%
```

## Baseline study

155 epochs:

```
Train set: Average loss: 0.0428, Accuracy: 98.69%
Test set: Average loss: 0.0884, Accuracy: 97.21%
```

https://github.com/anhinga/synapses/blob/master/Baseline_1.ipynb

718 epochs:

```
Train set: Average loss: 0.0046, Accuracy: 99.87%
Test set: Average loss: 0.1198, Accuracy: 97.60%
```

https://github.com/anhinga/synapses/blob/master/Baseline_1_1.ipynb

First experiment, 1000 epochs:

```
Train set: Average loss: 0.0025, Accuracy: 99.96%
Test set: Average loss: 0.1262, Accuracy: 97.74%
```

https://github.com/anhinga/synapses/blob/master/Baseline_complete.ipynb

`Baseline_complete` also contains the run of the last triplet of the original experiments with our new instrumentation reproducing a well pronounced negative learning at the last experiment (note that the last experiment in the triplet continues smoothly from the preceeding one, rather than starting at the beginning, hence the look of the trend curves).

I replaced

```python
for epoch in range(1, epochs + 1):
    show_MNIST_connections(evol_net)
    evolnet_history = train(log_interval, evol_net, device, train_loader, optimizer, epoch, evolnet_history)
    evolnet_history = test(evol_net, device, test_loader, evolnet_history)
    evolnet_history.plot()
```

with

```python
for epoch in range(1, epochs + 1):
    # THIS COMMENT SHOULD NOT BE HERE: In the paper, evolutions occur on each epoch
    if epoch != 1:
        evolnet_history.plot()
    show_MNIST_connections(evol_net)
    if epoch != 1:
        print('Train set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
            evolnet_history.train_loss[epoch-2], 100. * evolnet_history.train_acc[epoch-2]))
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
            evolnet_history.val_loss[epoch-2], 100. * evolnet_history.val_acc[epoch-2]))
    evolnet_history = train(log_interval, evol_net, device, train_loader, optimizer, epoch, evolnet_history)
    evolnet_history = test(evol_net, device, test_loader, evolnet_history)
    time.sleep(10) # YOU MIGHT WANT TO CHANGE THIS NUMBER OR REMOVE THIS LINE
```

in the last triplet of experiments. We see that strong negative learning of network topology in the last experiment of the triplet demonstrated by Michael Klear in his original experiments reproduces nicely.

---

In the extra round of experiments

https://github.com/anhinga/synapses/blob/master/Extra.ipynb

I rerun the first experiment with 155 epoch to demonstrate that things don't reproduce precisely in the current setup, but stay in the same ballpark, obtaining

```
Train set: Average loss: 0.0433, Accuracy: 98.63%
Test set: Average loss: 0.0904, Accuracy: 97.30%
```

instead of

```
Train set: Average loss: 0.0428, Accuracy: 98.69%
Test set: Average loss: 0.0884, Accuracy: 97.21%
```

and I run the last triplet of experiments with 1e-3 regularization. That run was unremarkable, but demonstrated a shift from virtually no network topology learning to positive network topology learning in the first two experiments of the triplet, and a shift from strongly pronounced negative topology learning to virtually no network topology learning in the last experiment of the triplet, further confirming the main results of this study.

## Future work

There is a lot of room here for interesting experiments. The following directions of future work seem promising.

### Regularization experiments

It makes sense to continue looking for more favorable regularization coefficients and modes.

There is no requirement that `weight_decay` should stay constant in time, or that it should be the same for all layers. It would be interesting to try other forms of regularization, such as L1 or L1+L2.

In particular, given that we see underfitting from the very beginning with `weight_decay=1e-2`, and that we still seem to get a bit of overfitting further down the line with `weight_decay=1e-3`, it seems promising to try the regularization mode with `weight_decay` gradually increasing as training goes on.

### Changes in evolutionary scheme

Things which might be considered if one wants to tighten on a particular solution include gradual decrease in the fraction of replaced edges. One should also consider exploring gradual tightening of sparsity (can we learn a **very sparse** topology here?).

This is an evolutionary scheme, where the absolute value of a weight serves as a fitness criterion for selection. However, the next generation is created completely randomly (no parents).

It might be interesting to create a fraction of the next generation from parents. For example, one can create some of the new edges closely to the existing ones, and to bias the choice of parents towards those with largest absolute values of weights. (Eventually, crossover (using multiple egdes to create the new ones in the vicinity) might be considered, but that, I think, is more optional/further down the road.)

### Recurrent networks and more interesting cases of topology learning

Applying this neuroevolutionary training scheme to various forms of recurrent networks is of interest.

Recently, Shiwei Liu, Mocanu, and Pechenizkiy conducted research into using this scheme to produce Intrinsically Sparse LSTMs (https://arxiv.org/abs/1901.09208 ; no open source code yet, as far as I know).

It would be interesting to try to do this with less structured varieties of recurrent networks, e.g. with Recurrent Identity Networks, https://arxiv.org/abs/1801.06105 (see also my comment, Understanding Recurrent Identity Networks, http://www.cs.brandeis.edu/~bukatin/recurrent-identity-networks.html )

It might be of particular interest if one focuses on learning the network topology (LSTMs already come with quite a bit of predefined topological structure, while, for example, Recurrent Identity Networks don't have much a priori structure, but merely a better regularization setup compared to naive vanilla RNNs).

### Acceleration of training, possibly including complete rewrites of the system using existing sparse matrix implementations

One might, for example, consider using "PyTorch Extension Library of Optimized Autograd Sparse Matrix Operations", https://github.com/rusty1s/pytorch_sparse

This has good potential for making things faster.

For this particular problem one would use the Sparse-Dense matrix multiplication, but for more elaborate setups such as as Graph Neural Networks or Dataflow Matrix Machines (https://github.com/jsa-aerial/DMM) one would typically use Sparse-Sparse matrix multiplication.


