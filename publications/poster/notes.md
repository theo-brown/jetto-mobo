# Poster notes

## 4 key points
- We demonstrate the application of **statistical modelling** and **machine learning**...
- ... to **design of experiments**
- We find that these methods are **much faster** than previous approaches...
- ... and allow **deeper analysis** of the tradeoffs between objectives

## Mini-speech
### Intro + BayesOpt
Our application is the design of electron cyclotron heating, with the goal of achieving good safety factor profiles.
The safety factor is a measure of the magnetic field's helicity.
It's been shown to have a variety of effects on the stability and performance of the plasma.

Many elements of reactor design are an iterative optimisation process.
You come up with a design, simulate it, and use the simulation to improve the design.
You cycle through this loop a few times - Design, Simulate, Evaluate.
The issue is that simulating designs in detail is very expensive, which means that trying out lots of designs isn't an option.

So what we do is we apply a Bayesian - probabilistic, belief-based - framework to the selection of candidate points.
The idea is that if you choose the points you want to simulate very carefully, you have to simulate fewer points.
This shifts some of the computational burden from the simulation step to the design step.
We found that this exhibits vastly improved performance over stochastic search methods, such as genetic algorithms.

BayesOpt is a model-based optimisation framework.
We train a statistical model that learns the mapping from inputs to objective value.
In this example case, the model predicts the 'score' of the safety factor profile directly from the ECRH parameters.
This is an important distinction - we're not predicting the q profile, instead we're predicting a vector that represents how good the q profile is.

Once we've fit the model, we then use it to pick the input points to try next.
We use its predictive capacity to select the candidates that we think will perform well based on what we've already seen.
The key thing here is that information about every past trial point is used in choosing the next point to try.

### Pareto optimality
In tokamak design scenarios we're interested in a lot of different objectives.
Often these objectives will clash with each other.
So for example, if you're wanting to go to the shop, there might be one that's really close to you - but it's very expensive - and one that's really far away that's very cheap.
Pareto optimality is a way of formulating solutions to this kind of problem.
The idea is that we find the set of points that are the best tradeoffs you can achieve - so you can't move from this point without compromising performance in one of your objectives.
This is the multi-dimensional extension of single-objective optimisation.

So going back to the optimisation method, the acquisition function chooses points that are likely to be Pareto optimal.

### Results
We try this with a couple of different ECRH parameterisations, and end up with interesting results.

Firstly, it's important to note that just because something's Pareto optimal it doesn't mean that it's desirable in practice - such as this orange one here.
Additional post-processing or constraints on objectives are required to only return the useful results.

The more interesting results are on the left.
Here, we use a piecewise linear ECRH function - this is the same kind that is currently used in STEP. 
We find two families of solutions - monotonic ones (blue) and double-peaked ones (yellow, red).
These double-peaked profiles had been theoretically predicted but stable solutions hadn't been found using previous methods.
Our method has managed to perform a more comprehensive search of the design space.

The second result is that we can now understand what the double peak actually achieves.
Looking at the radar plot, the yellow profile has pushed the minimum closer to the axis, at the expense of a smaller margin to q=2 and slightly reduced monotonicity.
So now we can make informed design decisions - choosing which of these objectives we want to weight more heavily will affect whether we want a monotonic or a double-peaked profile.
