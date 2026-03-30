import numpy as np
from __future__ import annotations
from typing import Union, Optional, List
class Value:
    def __init__(self, value: float):
        self.data = value
        self.grad = 0
        self._operation = 'leaf'
        self.out_edges = 0
        self._prev = []

    def __add__(self, other: Value):        
        new_value = Value(self.data + other.data)
        new_value._operation = 'add'
        new_value._prev = [self, other]
        self.out_edges += 1
        other.out_edges += 1

        return new_value
    
    def __sub__(self, other: Value):
        new_value = Value(self.data - other.data)
        new_value._operation = 'sub'
        new_value._prev = [self, other]
        self.out_edges += 1
        other.out_edges += 1

        return new_value
    
    def __truediv__(self, other: Value):
        new_value = Value(self.data / other.data)
        new_value._operation = 'div'
        new_value._prev = [self, other]
        self.out_edges += 1
        other.out_edges += 1
        return new_value

    def relu(self):
        new_value = Value(max(0, self.data))
        new_value._operation = 'relu'
        new_value._prev = [self]
        self.out_edges += 1
        return new_value

    def __mul__(self, other: Value):
        new_value = Value(self.data * other.data)
        new_value._operation = 'mul'
        new_value._prev = [self, other]
        self.out_edges += 1
        other.out_edges += 1
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
            self._prev[1].grad += self.grad * self._prev[0].data / (self._prev[1].data * self._prev[1].data)
        elif self._operation == 'relu':
            if self._prev[0].data >= 0:
                self._prev[0].grad += self.grad
        else:
            raise ValueError(f'The operation self._operation: {self._operation} is not implemented')

    

    def backward(self):
        # Topological sort and then backpropagation
        self.grad = 1

        mp = {}
        ready = set()
        ready.add(self)
        topo = []
        while len(ready):
            nxt = ready.pop()
            topo.append(nxt)
            for prev in nxt._prev:
                if mp.get(prev, -1) != -1:
                    mp[prev] -= 1
                    if mp[prev] == 0:
                        ready.add(prev)
                else:
                    mp[prev] = prev.out_edges - 1
                    if mp[prev] == 0:
                        ready.add(prev)
        print(f'graph size: {len(topo)}')
        for node in topo:
            node._backward()

class Neuron(Value):
        
    def __init__(self, neurons_previous_layer, weights = None):
        super().__init__(0)
        self._operation = 'neuron'
        self.in_layer = neurons_previous_layer
        if weights == None:
            weights_np = np.random.randn(len(self.in_layer)) * np.sqrt(2 / len(self.in_layer)) 
            self.weights = [Value(float(w)) for w in weights_np]
        elif isinstance(weights[0], Value):
            self.weights = weights
        else:
            self.weights = [Value(w) for w in weights]
        self._prev = neurons_previous_layer + self.weights
        self._prev.append(Value(0)) # Bias
        self.bias = self._prev[-1]

        self.data = 0
        for i in range(len(self.in_layer)):
            self.data += self.in_layer[i].data * self.weights[i].data
        self.data += self.bias.data

        for w in self.weights:
            w.out_edges += 1
        for neuron in self.in_layer:
            neuron.out_edges += 1
        self.bias.out_edges += 1

    def __add__(self, other: Value):
        raise NotImplementedError("Invalid operation over neurons")
    def __sub__(self, other: Value):
        raise NotImplementedError("Invalid operation over neurons")
    def __truediv__(self, other: Value):
        raise NotImplementedError("Invalid operation over neurons")
    def relu(self, other: Value):
        raise NotImplementedError("Invalid operation over neurons")
    def __mul__(self, other: Value):
        raise NotImplementedError("Invalid operation over neurons")
    
    def _backward(self):
        if self.data >= 0: # Relu gradient: 1 if argument non-negative
            print('activated')
            for i in range(len(self.in_layer)):
                self.in_layer[i].grad += self.grad * self.weights[i].data
            for i in range(len(self.weights)):
                self.weights[i].grad += self.grad * self.in_layer[i].data
            self.bias.grad += self.grad
        

class fc_layer:
    def __init__(self, size: int, previous_layer: Union[List[Value], fc_layer]):
        self.neurons: List[Value] = []
        if isinstance(previous_layer, fc_layer):
            self.neurons = [Neuron(previous_layer.neurons) for i in range(size)]
        else:
            self.neurons = [Neuron(previous_layer) for i in range(size)]
        
class mlp:
    def __init__(self, in_dim: int, neurons_per_layer: List[int]):
        self.initial_layer = [Value(0) for i in range(in_dim)]
        self.layers = [fc_layer(neurons_per_layer[0], self.initial_layer)]
        for layer in neurons_per_layer[1:]:
            self.layers.append(fc_layer(layer, self.layers[-1]))

    def backward(self):
        for neuron in self.layers[-1]:
            neuron.backward()




A = Value(-3)
B = Value(2)
C = Value(10)
neuron = Neuron([A, B, C])

neuron.backward()

print(f'{A.grad}, {B.grad}, {C.grad}, {neuron.grad}')
print([weight.grad for weight in neuron.weights])
print(neuron.bias.grad)
