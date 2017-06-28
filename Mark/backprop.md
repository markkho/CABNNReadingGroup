Backprop in a nutshell
======================

Backprop is gives us a way to calculate the effect of different parameters in
a network on a cost function (i.e. their gradients). We can then modify those 
parameters to reduce the cost of the network.

Basic Neural Net Architecture
-----------------------------
The zeroth layer of the network is a vector, $\mathbf{x}$. The _**input sum**_ 
of the first (hidden) layer is a weighted combination of these inputs plus a 
bias:
$$
\mathbf{z}^0 = \mathbf{W}^0\mathbf{x} + \mathbf{b}^0.
$$
The activation of the first layer is an element-wise sigmoid function of the 
input sum:
$$
\mathbf{a}^0 = \sigma(\mathbf{z}^0) = \frac{1}{1+e^{-\mathbf{z}^0}}.
$$

More generally, the input sum for the $k$-th node in layer $l$, $z^l_k$ is a 
function of the activations from the previous layer ($\mathbf{a}^{l-1}$), 
the connecting weights ($\mathbf{w}^l_k$), and a bias ($b^l_k$):
$$
z^l_k = \mathbf{a}^{l-1} \cdot \mathbf{w}^l_k + b^l_k.
$$
The node then outputs an activation $\mathbf{a^{l+1}_k}$ according to a sigmoid 
function:
$$
\mathbf{a^{l+1}_k} = \sigma(z^l_k) = \frac{1}{1+e^{-z^l_k}}.
$$
This activation in turn affects the input sums in the next layer of nodes, 
$\mathbf{z}^{l+1}$.

In the last layer, $L$, the network will produce an output activation 
that will be evaluated with a differentiable cost function $C(\mathbf{a}^L)$.

Derivation of Backprop
-------------------
For any given weight $w^l_{kj}$, we would like to calculate its effect on the 
final cost $C$. That is, we want:
$$
\begin{aligned}
\frac{\partial C}{\partial w^l_{kj}} & = 
    \frac{\partial z^l_k}{\partial w^l_{kj}}
    \frac{\partial C}{\partial z^l_k} \\
& = \frac{\partial z^l_k}{\partial w^l_{kj}}
    \frac{\partial a^l_k}{\partial z^l_k}
    \frac{\partial C}{\partial a^l_k}
\end{aligned}
$$
The derivative of the input sum w.r.t. the weight is straightforward 
to compute:
$$
\begin{align}
\frac{\partial z^l_k}{\partial w^l_{kj}} & = 
    \frac{\partial}{\partial w^l_{kj}} 
    \mathbf{a}^{l-1} \cdot \mathbf{w}^l_k + b^l_k \\
& = a^{l-1}_{kj}
\end{align}
$$
As is the derivative of the activation w.r.t. the input sum:
$$
\begin{align}
\frac{\partial a^l_k}{\partial z^l_k} & = 
    \frac{\partial}{\partial z^l_k} \sigma(z^l_k) \\
& = \frac{\partial}{\partial z^l_k} \frac{1}{1+e^{-z^l_k}} \\
& = (\frac{1}{1+e^{-z^l_k}})(1 - \frac{1}{1+e^{-z^l_k}}) \\
& = a^l_k(1 - a^l_k)
\end{align}
$$

A bit more complicated is deriving the final term since the activation, 
$a^l_k$, affects the cost through multiple nodes at the $l+1$-th layer. For 
this we need to use the [chain rule](http://www.solitaryroad.com/c353.html):
$$
\begin{align}
\frac{\partial C}{\partial a^l_k} & = 
    \frac{\partial}{\partial a^l_k} C(\mathbf{z}^{l+1}) \\
& = \frac{\partial C}{\partial z^{l+1}_{1}}
    \frac{\partial z^{l+1}_{1}}{\partial a^l_k} + ... + 
    \frac{\partial C}{\partial z^{l+1}_{M}}
    \frac{\partial z^{l+1}_{M}}{\partial a^l_k} \\
& = \sum_{m}\frac{\partial C}{\partial z^{l+1}_{m}}
    \frac{\partial z^{l+1}_{m}}{\partial a^l_k} \\
& = \sum_{m}\frac{\partial C}{\partial z^{l+1}_{m}}w^{l+1}_{mk}
\end{align}
$$
We can denote $\frac{\partial C}{\partial z^{l+1}_{m}}$ the _**error**_ at 
neuron $m$ in layer $l+1$ or $\delta^{l+1}_m$. This allows us to write the 
partial derivative as:
$$
\frac{\partial C}{\partial a^l_k} = \sum_{m}\delta^{l+1}_m w^{l+1}_{mk}.
$$
Thus:
$$
\frac{\partial C}{\partial w^l_{kj}} = 
    a^{l-1}_{kj} a^l_k(1 - a^l_k) 
    \bigg(\sum_{m}\delta^{l+1}_m w^{l+1}_{mk}\bigg)
$$