### MicroTorch Library
Library of PyTorch implementations of microstructural and quantitative MRI models.

Cardiff University Brain Research Imaging Centre

## Dependencies (incomplete)
PyTorch
Numpy

## Command line examples

*Ball-stick (doesn't quite work)*

python3 fit.py -img data.nii.gz -ma mask.nii.gz -bvals bvals.txt -bvecs bvecs.txt -d 24 -sd 8 -se 123 -m BallStick -a relu -lr 0.0001 

*MSDKI (works but this model doesn't seem to like dropout)*

python3 fit.py -img data.nii.gz -ma mask.nii.gz -bvals bvals.txt -bvecs bvecs.txt -d 24 -sd 8 -se 123 -m MSDKI -a elu -lr 0.01  
