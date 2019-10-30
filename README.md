# Damped oscillation parameter detector

![example](https://github.com/dariusarnold/damped_oscillation_bestfit/blob/master/example.png "Example of program output")

Read seismogram from SEG-2 ASCII file, graphically select a part of it and fit a dampened oscillation ![equation](https://latex.codecogs.com/svg.latex?x%28t%29%20%3D%20A%20*%20exp%28-%5Cdelta%20t%29*sin%28%5Cphi%20&plus;%20%5Comega%20t%29) to it to find amplitde A, damping constant delta, phase phi and frequency omega of the oscillation.  

## How to use

Pass path to SEG-2 ASCII file as first command line argument.

## Requirements

 - Python 3.5 or greater
 - numpy
 - scipy
 - matplotlib
