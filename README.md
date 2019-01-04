# Markov Chain Estimator for Star Formation History
Estimate cumulative star formation history median distribution from
hybridMC produced star formation history (SFR per age bin).

Requires "isochrones" directory for stellar evolution model endpoints.

Usage:

1. For a single file:
```./sfh_to_prob.py $PATH/fit_XYZ.complete -nbins 22 -n 500000```

Use -nbins 22 for 50 Myr, -nbins 26 for 80 Myr
Increase -n value for larger randm sampling if job fails
These two parameters must be integers, e.g., "-nbins 22.0" oe
"-n 1e7" will fail. 

2. For mutiple files, mute screen output and produce latex tables:
```ls $PATH/*/*.complete | xargs ./sfh_to_prob.py -nbins 22 -n 500000 -silent -latex```
