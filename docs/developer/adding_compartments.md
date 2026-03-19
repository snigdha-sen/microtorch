## Adding a New Compartment

New microstructure compartments can be added by adding a new class to one of the following files in `src/signal_models`:

```
src/signal_models/gaussian_models.py
src/signal_models/sphere_models.py
src/signal_models/cylinder_models.py
src/signal_models/distributed_models.py
```
Since multi-compartment models are identified by MicroTorch using PascalCase, **each compartment name must begin with a capital letter and be followed only by lowercase letters and numbers.**

For example, `Compartment` and `Compartmentt2` are valid, whereas `COMPARTMENT`, `FancyCompartment`, and `CompartmentT2` are not.

Each compartment must follow the structure below:

``` python
class Compartmentname:
    def __init__(self):
    	self.parameter_ranges = [[min_A, max_A], ...]  # Acceptable range for each parameter
     	self.parameter_names = ['A', ...]   # Parameter names
		self.n_parameters = N      # Number of parameters                         
		
		# Whether this compartment model represents the spherical mean of the signal. 
		# Options:
		# True - spherical mean models (e.g. Astrosticks)
		# False - directional compartments (e.g. Stick)
		# None - usable in either spherical mean or directional models (e.g. Ball) 
 		self.spherical_mean = True                   

    def __call__(self, grad, params):

        # Acquisition parameters
        # The Grad class provides:
        # b_values, b_vecs, Delta, delta, gradient_strength, TE, bdelta
        ac_param = grad.ac_param

        # Extract parameters (i corresponds to index in parameter_names and parameter_ranges)
        param_A = params[:, i].unsqueeze(1)

        # Signal equation (must combine pytorch functions such that equation is fully differentiable)
        S = ...

        return S
```

### Requirements

-   The forward model must be **fully differentiable** (compatible with
    PyTorch autograd).
-   Parameter ordering must match `parameter_names`.
-   Parameter ranges should reflect physically meaningful bounds.
-   Output shape must match the expected signal shape.

### Other files you need to change when adding a compartment

In addition, when adding a new compartment you should:

- Add the compartment name **in both required locations** in `src/signal_models/__init__.py`.
- Add an example gradient file suitable for estimating the compartment’s parameters to `simulation_data/grad/` (if one does not already exist).
- Register the model and its example gradient file in `src/utils/make_test_image.py`.
- Optionally, add appropriate parameter ranges or fixed parameters via a yaml file (see below).