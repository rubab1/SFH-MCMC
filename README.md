# Markov Chain Estimator for Star Formation History
MCMC analysis of binned star formation rates to statistically
estimate probabilistic star formation history.
Examples:

```
./sfh_to_prob.py dir3/fit_gst311b.err -nbins 22 -n 500000 -silent -latex
ls dir?/*.err | xargs ./sfh_to_prob.py -nbins 22 -n 500000 -latex > SFH_Tables.tex
```
