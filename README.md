# Basic Neural Network

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/rizgiak/basic_neural_network/)

### Installation
This package using `gnuplot` to plot the error graph and `gnuplot-iostream` as include
```sh
$ sudo apt-get install gnuplot
```
You can clone `gnuplot-iostream` in
```sh
$ git clone https://github.com/dstahlke/gnuplot-iostream.git
$ cd gnuplot-iostream
$ make
```

Copy file `gnuplot_iostream.h` to this package directory

Compile this package with
```sh
$ g++ -o BasicNeuralNetwork BasicNeuralNetwork.cpp -lboost_iostreams -lboost_system -lboost_filesystem
```
