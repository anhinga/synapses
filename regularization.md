**work in progress; should be done by March 4, 2019**

# Regularization in intrinsically sparse networks

In 2017, Mocanu et al. pubslihed a remarkable neuroevolutionary scheme for training sparse neural nets (arxiv: 1707.04780, then appearing in Nature Communications **9** (19 June 2018), open source repository: https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks )

One starts with initializing sparse layers with random connectivity and then trains by repeating the following 2-step cycle a number of times: do some training, delete a fraction of connections with weights closest to zero, and recreate random new connections instead of the deleted ones.

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

We will call the network topology learning demonstrated by Mocanu et al **positive learning**, and we will call the inverse pattern of the network topology learning emerging in the runs performed by Michael Klear **negative learning**. ("Negative" here does not a priori imply "bad", although as we will see below, in this series of experiments negative learning is usually associated with some overfitting/failure to generalize.)

The main conjecture I made was that this effect was related to the absence of regularization in the original code. The logic I followed was that in the absence of regularization, when the weights pointing from the outlying areas are created, they tend to remain unchanged by training. At the same time, meaningful connections are changed by training, and occasionally become small and therefore get eliminated more frequently.

At the same time, if one were to add a sufficiently strong regularization encouraging smaller weights, then one would expect the connections which are not informative to the result to decrease on average more rapidly, than the connections which are informative.

The experiments we performed here and are describing below seem to confirm this conjecture. Namely, when one adds sufficiently strong reglarization, negative learning is replaced by positive learning, and the stronger regularization is, the more pronounced is this effect.

---


[...]


## Details of experiments with regularization

The first experimental notebook confirmes the conjecture stated in that issue:

https://github.com/anhinga/synapses/blob/master/Experiment_1.ipynb

We also see that this level of regularization makes the model generalize nicely, while without regularization it tends to somewhat overfit, cf. the original Jupiter notebook:

https://github.com/anhinga/synapses/blob/master/MNIST_demo.ipynb

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



## Baseline study

## Future work

**This is currently work in progress, with further software experiments currently running. Here is a longer version of experiment_1:**

https://github.com/anhinga/synapses/blob/master/Experiment_1_complete.ipynb

**Here is a similar run with 1e-2 regularization (which seems to learn the topology even better, but converges way too slowly):**

https://github.com/anhinga/synapses/blob/master/Experiment_2.ipynb

**I am going to do a couple of additional runs after that. And I am also going to edit this write-up - I expect to be mostly done with all this by March 4, 2019.**
