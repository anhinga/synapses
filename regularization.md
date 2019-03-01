**work in progress; should be done by March 4, 2019**

# Regularization in intrinsically sparse networks


This fork was originally created to investigate a strange effect described in

https://github.com/AlliedToasters/synapses/issues/1

## Conjectures and high-level description of experimental findings


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
