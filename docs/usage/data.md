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

For example, the package includes an example Human Connectome Project grad file 
`src/microtorch/resources/protocols/grad_HCP_with_deltas.txt`:

```text
# gx         gy         gz         b      Δ      δ
0.906387   0.281846   0.314683   1000   43.1   10.6
-0.395044  0.826284   0.401490   1995   43.1   10.6
0.149635   0.488142  -0.859841   3005   43.1   10.6
0.177856  -0.927837   0.327849    995   43.1   10.6
...
```

**OR** specify acquisition parameters individually using FSL-style files:

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

Only the acquisition parameters required by the selected signal model need to be provided. For example, many diffusion MRI models require only a four-column MRtrix-style gradient file (gradient directions and b-values) or, equivalently, FSL-style bvecs and bvals files.