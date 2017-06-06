Notes
=====
TensorFlow concepts
-------------------
- A "Tensor" is symbolic, it doesn't actually hold values but needs to be
 run. It can be passed as input into another operation that can get run.



Math concepts
-------------
- Cross Entropy:
$H(p, q) = E_p [-log(q)] = H(p) + D_{KL}(p || q)$