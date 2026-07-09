## Testing with Simulated Data

To verify that everything is configured correctly and to test model fitting using data with known ground truth, you can generate synthetic test images for all currently defined models and compartments, then fit models to this data.


## A. Generate synthetic data

You can generate simulated test data with:

```bash
microtorch-create-test-images
```

The images are created using example gradient files stored in:

```
INSTALL_DIR/microtorch/src/microtorch/resources/protocols 
```

The generated datasets will be saved in 

`INSTALL_DIR/microtorch/simulation_data/data`.

where `INSTALL_DIR` is your chosen installation location.

## B. Fit models to synthetic data 

To fit models to all of the simulated datasets, run:

```bash
microtorch-create-test-images --fit
```

The fitted parameter maps will be written to 

`INSTALL_DIR/microtorch/outputs`

## C. Assess fits 


You can then compare the fitted parameter values with the ground truth simulation parameters using the notebook:

```
INSTALL_DIR/microtorch/examples/plot_test_images.ipynb
```

This notebook compares the fitted parameters against the ground truth values to assess if fits are working as expected.