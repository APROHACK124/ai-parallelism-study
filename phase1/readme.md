# Phase 1

In this phase, I implement Autograd, similar to  [pytorch's autograd engine](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), starting with a `Value` class that represent the functions that I keep track on, defining various simple operations such as addition, multiplication, and relu to allow us to create complex functions.

Using this class it's just a matter of calling `Value.backward()` to calculate the gradient of that value with respect to all of the functions from which the value depends on.

This is done by doing a topological sort over the dependency graph, and iterating from the end to the start, "pushing" the gradient to the back depending on the function involved at each edge.

Then I implemented a Neuron class that uses Values to maintain the weights and bias of a classic perceptron, from there a fully connected layer, and then a mlp class, all of which reassemble `torch.nn.module` subclasses.

At `notebook.ipynb` I run some tests over this implementation, basically creating a mlp with my implementation of autograd, and training over a `sklearn` simple dataset. In the second plot we can see that the model is actually learning something.

Then I do some checks to verify that the gradients are properly calculated, using approximations, and then I check that the model is able to memorize examples if trained with just a few.
