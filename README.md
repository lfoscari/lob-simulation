## `simulation.py`

Run a simulation of the limit-order-book market for a given number of rounds and produce several plots. By editing the code you can fix an inflationary and a non-inflationary strategy, then run either one with the first argument (infl | not-infl). Optionally you can provide the name of a pdf where the script will save the results

See `inflationary.pdf` and `not-inflationary.pdf` for examples.

## `learning-simulation.py`

Similar to the previous one, but now the starting strategy is changed during play using a collaborative and a competitive approach, the former aims at maximising the price, while the former at reducing it. The simulation is repeated a number of times and the results are averaged. To improve the visualization, the log-values are displayed on a linear plot.

See `learning-simulation.pdf` for an example.

## `collusion-volume.py`

Draw random values for the parameters and display the proportion with positive mu.
