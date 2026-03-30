from __future__ import annotations
import numpy as np
from typing import Union, Optional, List, Tuple
class Value:
    def __init__(self, value: float):
        self.data = value
        self.grad = 0
        self._operation = 'leaf'
        
        self._prev = []

    def __add__(self, other: Value):        
        new_value = Value(self.data + other.data)
        new_value._operation = 'add'
        new_value._prev = [self, other]

        return new_value
    
    def __sub__(self, other: Value):
        new_value = Value(self.data - other.data)
        new_value._operation = 'sub'
        new_value._prev = [self, other]

        return new_value
    
    def __truediv__(self, other: Value):
        new_value = Value(self.data / other.data)
        new_value._operation = 'div'
        new_value._prev = [self, other]
        return new_value

    def relu(self):
        new_value = Value(max(0, self.data))
        new_value._operation = 'relu'
        new_value._prev = [self]
        return new_value

    def __mul__(self, other: Value):
        new_value = Value(self.data * other.data)
        new_value._operation = 'mul'
        new_value._prev = [self, other]
        return new_value

    def _backward(self):
        if self._operation == 'leaf':
            return 
        if self._operation == 'add':
            for val in self._prev:
                val.grad += self.grad
        elif self._operation == 'mul':
            self._prev[0].grad += self._prev[1].data * self.grad
            self._prev[1].grad += self._prev[0].data * self.grad
        elif self._operation == 'sub':
            self._prev[0].grad += self.grad
            self._prev[1].grad -= self.grad
        elif self._operation == 'div':
            self._prev[0].grad += self.grad / self._prev[1].data
            self._prev[1].grad -= self.grad * self._prev[0].data / (self._prev[1].data * self._prev[1].data)
        elif self._operation == 'relu':
            if self._prev[0].data >= 0:
                self._prev[0].grad += self.grad
        else:
            raise ValueError(f'The operation self._operation: {self._operation} is not implemented')
    
    def zero_grad(self):
        self.grad = 0

    

    def backward(self):
        # Topological sort and then backpropagation
        
        # DFS

        visited = set()
        topo: List[Value] = []

        def dfs(node: Value):
            visited.add(node)
            for prev in node._prev:
                if prev in visited:
                    continue
                dfs(prev)
            topo.append(node)

        dfs(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
        
    

class Neuron:
    def __init__(self, nin, is_linear: bool):
        self.w = [Value(float(np.random.randn() * np.sqrt(2 / nin))) for _ in range(nin)]
        self.bias = Value(0.0)
        self.is_linear = is_linear

    def __call__(self, x):
        ret = self.bias
        for wi, xi in zip(self.w, x):
            ret += wi * xi
        if not self.is_linear:
            return ret.relu()
        return ret
    def zero_grad(self):
        for wi in self.w:
            wi.zero_grad()
        self.bias.zero_grad()

    def parameters(self) -> List[Value]:
        return self.w + [self.bias]

    
class fc_layer:
    def __init__(self, nin, number_neurons, is_linear: bool):
        self.neurons = [Neuron(nin, is_linear) for _ in range(number_neurons)]
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out
    
    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

    def parameters(self) -> List[Value]:
        param: List[Value] = []
        for neuron in self.neurons:
            param = param + (neuron.parameters())
        return param
    



class mlp:
    def __init__(self, nin: int, layers: List[int], layer_types: List[str]):
        all_layers = [nin] + layers
        self.layers = [fc_layer(all_layers[i - 1], all_layers[i], layer_types[i - 1] == 'Linear') for i in range(1, len(all_layers))]

    def __call__(self, x):
        to_value = [Value(i) for i in x]
        for layer in self.layers:
            to_value = layer(to_value)
        return to_value
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self) -> List[Value]:
        # Returns a list of the (trainable) values
        param = []
        for layer in self.layers:
            param = param + layer.parameters()
        return param
