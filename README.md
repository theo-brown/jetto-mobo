<h1 align="center">
    jetto-mobo
</h1>
<p align="center">
    <em>
        Multi-objective Bayesian optimisation for tokamak design in the JETTO plasma modelling code
    </em>
</p>

This repository is currently under active development. It forms one part of my master's thesis on data-driven methods for tokamak control and design.

### Notes

#### Environment
Conda:
- Python >= 3.7
- hdf5
Pip:
- all packages in requirements.txt

#### Singularity
To build JINTRAC Singularity image from docker tgz
```
singularity build sim.v220922.sif docker-archive://sim.v220922.tgz
```
