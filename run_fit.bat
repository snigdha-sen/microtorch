@echo off

python -m microtorch.fit ^
  -mod Standard_wm ^
  -img data_sim_abstract\recon_signal_lmax2_new.nii.gz ^
  -bvals data_sim_abstract\bvals_grad_prisma_new.bval ^
  -bvecs data_sim_abstract\bvecs_grad_prisma_new.bvec ^
  -bd data_sim_abstract\bdelta_grad_prisma_new.bdelta ^
  -lss 1000 ^
  -nl 3 ^
  -ma data_sim_abstract\wm_mask_011.nii.gz ^
  -ni 30