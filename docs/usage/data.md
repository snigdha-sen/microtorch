# Data and Acquisition Parameters

## Data and File Paths

### Required

``` bash
data.image=/path/to/dwi.nii
```

Optional mask:

``` bash
data.mask=/path/to/mask.nii
```

## Acquisition Parameters

You can either provide a single gradient scheme file:

``` bash
acquisition.grad=/path/to/grad.scheme
```

where gradient scheme files use an extended MRtrix-style format.

- Columns 1–3: Gradient direction vector (x, y, z)
- Column 4: b-value [ms/μm²]
- Column 5: Diffusion gradient separation, Δ (“big delta”) [ms]
- Column 6: Diffusion gradient duration, δ (“small delta”) [ms]
- Column 7: Echo time (TE) [ms]

**OR** specify acquisition parameters individually:

``` bash
acquisition.bvals=/path/to/bvals
acquisition.bvecs=/path/to/bvecs
acquisition.delta=/path/to/delta
acquisition.smalldelta=/path/to/smalldelta
acquisition.TE=/path/to/TE
acquisition.TR=/path/to/TR
acquisition.TI=/path/to/TI
acquisition.bdelta=/path/to/bdelta
```

You only need to provide the parameters required by your selected model.