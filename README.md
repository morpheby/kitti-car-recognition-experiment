
## Morphe-µ project ##

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This project was intended to test a hypothetical neural network architecture, and it failed. You can read
through the whole experimenting process in [mm-neural-class1-study.ipynb](mm-neural-class1-study.ipynb).

I post this project with the hopes that maybe someone will find this useful.

## Short note on the proposed NN structure ##

The core idea was to implement an MLP that would output weights for another MLP, and those provide
better (and presumably, faster converging) fitting. Unfortunately, all attempts at modeling this architecture
turned out to be either of almost same performance as usual MLP, or worse, and they require much
more memory and processing power to be trained.

For more information, read [mm-neural-class1-study.ipynb](mm-neural-class1-study.ipynb).
