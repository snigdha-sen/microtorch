## Testing with Synthetic Data

To verify that everything is configured correctly and to test model fitting using data with known ground truth, you can generate synthetic test images for all currently defined models and compartments:

```bash
./scripts/create_test_images.py
```

The generated test datasets will be saved in:

```
simulation_data/data
```

The images are created using example gradient files stored in:

```
simulation_data/grad
```

---

To automatically run model fitting on each of the generated test datasets:

```bash
./scripts/create_test_images.py --fit
```

---


To assess the quality of the fits, open and run:

```
examples/plot_test_images.ipynb
```

This notebook compares the fitted parameters against the ground truth values to assess if fits are working as expected.