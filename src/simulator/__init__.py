"""Forward model of a confocal fluorescence microscope.

Split into:
  - io           : loading/parsing (TIFF stacks, DLS xlsx, dark frames)
  - estimation   : microscope parameter estimation (gain, PSF, backgrounds)
  - forward_model: the generative simulator (simulate_* + PMT noise)
"""
