# cleverspeech.graph

This houses all the classes needed to make an attack graph using the `Constructor` class.

### `Base.py` 
Factory class that consumes all other attack child classes.
Add elements in the order that class methods are listed 
(unless you just want to give yourself a headache). 

### `Graphs.py`
Create Tensorflow graphs of variable objects and placeholder to run an attack. 
Most of the magic happens here.

This is currently inherits the `Base.py` graph. 
But I want to pass this in independently for more freedom.    

### `Constraints.py`
Hard constraints applied to a perturbation. 
Required for graphs at the moment (won;t be in future).
Will be looking at various non Lp-norm metrics shortly.

TODO: The procedure shouldn't update tau/bound, classes here should. 
 
### `Lossses.py`
Various loss functions that can be used for adversarial objective minimisation functions.
Losses like `LowConf` and `HighConf` need an alignment generated from `Alignments.py` 
(they are based on the improved loss in Carlini & Wagner's Targeted Audio Attacks for ASR).

TODO: Base class to determine whether a loss is the non-linear classification constraint 
or a soft distance constraint regulariser.

### `Alignments.py`
Class that finds us a useful alignment for the confidence losses. 

### `Optimisers.py`
Some Tensorflow optimiser classes wrapped up so they will play nicely with the rest of the graph.

### `Procedures.py`
The actual procedure for running an attack. 
I only look at iterative attacks, so this is where a lot the magic happens.
This class needs to return the results as well so it's currently fairly large (and gets complicated quickly).

Many TODOs: Especially how we check for success, bound updates, generating SECs and many more!
