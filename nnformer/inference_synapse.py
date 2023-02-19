import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric

import scipy.ndimage
neighbour_code_to_normals = [
  [[0, 0, 0]],
  [[0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
  [[0.125, -0.125, 0.125]],
  [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
  [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
  [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
  [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
  [[0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
      [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
  [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
      [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
  [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
      [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
  [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
      [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
  [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
  [[0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
  [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
  [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
  [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
      [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
      [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
      [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
  [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
      [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
  [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
  [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
  [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
  [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
  [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
      [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
  [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
      [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
  [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
  [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
  [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
      [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
  [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
  [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
  [[-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
  [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
      [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
  [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
  [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
      [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
  [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
      [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
  [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
  [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
  [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
      [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
  [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
  [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
      [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
  [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
      [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
  [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
      [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
  [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
      [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
  [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
      [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
      [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
  [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
      [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
  [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
  [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
      [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
  [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
  [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
  [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
      [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
  [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
  [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
  [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
      [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
      [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
  [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
      [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
      [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
  [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
      [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
      [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
  [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
      [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
      [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
  [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
  [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
      [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
  [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
  [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
      [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
  [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
      [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
  [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
      [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
  [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
  [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
  [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
  [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
  [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0],
      [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
  [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
  [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
      [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
  [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
      [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
  [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
  [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
      [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
  [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
      [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
  [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
  [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
  [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
  [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
  [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
  [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
      [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
  [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
      [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
      [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
  [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
      [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
  [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
  [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
  [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
  [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
  [[0.125, -0.125, 0.125]],
  [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
  [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
      [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
  [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
      [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
  [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
      [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
  [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
  [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
      [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
  [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
  [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
  [[0.125, -0.125, -0.125]],
  [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
  [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
  [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
  [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
  [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
  [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
  [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
  [[-0.125, 0.125, 0.125]],
  [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
  [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
  [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
  [[0.125, -0.125, 0.125]],
  [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
  [[-0.125, -0.125, 0.125]],
  [[0.125, 0.125, 0.125]],
  [[0, 0, 0]]]


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """Compute closest distances from all surface points to the other surface.

    Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
    the predicted mask `mask_pred`, computes their area in mm^2 and the distance
    to the closest point on the other surface. It returns two sorted lists of
    distances together with the corresponding surfel areas. If one of the masks
    is empty, the corresponding lists are empty and all distances in the other
    list are `inf`

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.
      spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
          direction

    Returns:
      A dict with
      "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
          from all ground truth surface elements to the predicted surface,
          sorted from smallest to largest
      "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
          from all predicted surface elements to the ground truth surface,
          sorted from smallest to largest
      "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
          the ground truth surface elements in the same order as
          distances_gt_to_pred
      "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
          the predicted surface elements in the same order as
          distances_pred_to_gt

    """

    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return {"distances_gt_to_pred":  np.array([]),
                "distances_pred_to_gt":  np.array([]),
                "surfel_areas_gt":       np.array([]),
                "surfel_areas_pred":     np.array([])}

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    print("bounding box min = {}".format(bbox_min))
    print("bounding box max = {}".format(bbox_max))

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right and the back side. This is required to obtain the "full"
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                            bbox_min[1]:bbox_max[1]+1,
                                            bbox_min[2]:bbox_max[2]+1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                                bbox_min[1]:bbox_max[1]+1,
                                                bbox_min[2]:bbox_max[2]+1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                        [32, 16]],
                        [[8, 4],
                            [2, 1]]])
    neighbour_code_map_gt = scipy.ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = scipy.ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) &
                    (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the surface voxels)
    if borders_gt.any():
        distmap_gt = scipy.ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = scipy.ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(
            sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(
            sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {"distances_gt_to_pred":  distances_gt_to_pred,
                "distances_pred_to_gt":  distances_pred_to_gt,
                "surfel_areas_gt":       surfel_areas_gt,
                "surfel_areas_pred":     surfel_areas_pred}


def compute_average_surface_distance(surface_distances):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = np.sum(
        distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
    average_distance_pred_to_gt = np.sum(
        distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:
        surfel_areas_cum_gt = np.cumsum(
            surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[min(
            idx, len(distances_gt_to_pred)-1)]
    else:
        perc_distance_gt_to_pred = np.Inf

    if len(distances_pred_to_gt) > 0:
        surfel_areas_cum_pred = np.cumsum(
            surfel_areas_pred) / np.sum(surfel_areas_pred)
        idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[min(
            idx, len(distances_pred_to_gt)-1)]
    else:
        perc_distance_pred_to_gt = np.Inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    rel_overlap_gt = np.sum(
        surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
    rel_overlap_pred = np.sum(
        surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
    return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(
        surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
        np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    pancreas = label == 11

    return spleen, right_kidney, left_kidney,gallbladder,liver,stomach,aorta,pancreas


def test(fold):
    path = '/home/daniya.kareem/nnFormer/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_Synapse/'
    label_list = sorted(glob.glob(os.path.join(path, 'labelsTr','*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join('/home/daniya.kareem/Desktop/inferTs','*nii.gz')))
    list = ['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022',
            'label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

    label_list_sr = []
    for item in label_list:
        item = item.split('/')[-1]
        label_list_sr.append(item)

    label_lt = []
    for item in list:
        # item=item.split('/')[-1]
        # item =os.join(item,"*nii.gz")
        item = item+'.nii.gz'
        if item in label_list_sr:
            label_lt.append(item)
    label_list = []
    for i in range(len(label_lt)):
        a = os.path.join(path, 'labelsTr',label_lt[i])
        label_list.append(a)
    # label_list=sorted(glob.glob(label_list))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_spleen = []
    Dice_right_kidney = []
    Dice_left_kidney = []
    Dice_gallbladder = []
    Dice_liver = []
    Dice_stomach = []
    Dice_aorta = []
    Dice_pancreas = []

    hd_spleen = []
    hd_right_kidney = []
    hd_left_kidney = []
    hd_gallbladder = []
    hd_liver = []
    hd_stomach = []
    hd_aorta = []
    hd_pancreas = []

    nsd_spleen = []
    nsd_right_kidney = []
    nsd_left_kidney = []
    nsd_gallbladder = []
    nsd_liver = []
    nsd_stomach = []
    nsd_aorta = []
    nsd_pancreas = []

    file = path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_spleen, label_right_kidney,label_left_kidney,label_gallbladder,label_liver,label_stomach,label_aorta,label_pancreas = process_label(label)
        infer_spleen, infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_liver,infer_stomach,infer_aorta,infer_pancreas = process_label(infer)

        Dice_spleen.append(dice(infer_spleen, label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney, label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney, label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder, label_gallbladder))
        Dice_liver.append(dice(infer_liver, label_liver))
        Dice_stomach.append(dice(infer_stomach, label_stomach))
        Dice_aorta.append(dice(infer_aorta, label_aorta))
        Dice_pancreas.append(dice(infer_pancreas, label_pancreas))

        hd_spleen.append(hd(infer_spleen, label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney, label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney, label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder, label_gallbladder))
        hd_liver.append(hd(infer_liver, label_liver))
        hd_stomach.append(hd(infer_stomach, label_stomach))
        hd_aorta.append(hd(infer_aorta, label_aorta))
        hd_pancreas.append(hd(infer_pancreas, label_pancreas))

        nsd_spleen.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_spleen, label_spleen, [0.76,0.76,3]),4))
        nsd_right_kidney.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_right_kidney, label_right_kidney, [0.76,0.76,3]),4))
        nsd_left_kidney.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_left_kidney, label_left_kidney, [0.76,0.76,3]),4))
        nsd_gallbladder.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_gallbladder, label_gallbladder, [0.76,0.76,3]),4))
        nsd_liver.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_liver, label_liver, [0.76,0.76,3]),4))
        nsd_stomach.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_stomach, label_stomach, [0.76,0.76,3]),4))
        nsd_aorta.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_aorta, label_aorta, [0.76,0.76,3]),4))
        nsd_pancreas.append(compute_surface_dice_at_tolerance(compute_surface_distances(infer_pancreas, label_pancreas, [0.76,0.76,3]),4))

        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))

        fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))

        fw.write('nsd_spleen: {:.4f}\n'.format(nsd_spleen[-1]))
        fw.write('nsd_right_kidney: {:.4f}\n'.format(nsd_right_kidney[-1]))
        fw.write('nsd_left_kidney: {:.4f}\n'.format(nsd_left_kidney[-1]))
        fw.write('nsd_gallbladder: {:.4f}\n'.format(nsd_gallbladder[-1]))
        fw.write('nsd_liver: {:.4f}\n'.format(nsd_liver[-1]))
        fw.write('nsd_stomach: {:.4f}\n'.format(nsd_stomach[-1]))
        fw.write('nsd_aorta: {:.4f}\n'.format(nsd_aorta[-1]))
        fw.write('nsd_pancreas: {:.4f}\n'.format(nsd_pancreas[-1]))

        dsc = []
        HD = []
        nsd = []
        dsc.append(Dice_spleen[-1])
        dsc.append((Dice_right_kidney[-1]))
        dsc.append(Dice_left_kidney[-1])
        dsc.append(np.mean(Dice_gallbladder[-1]))
        dsc.append(np.mean(Dice_liver[-1]))
        dsc.append(np.mean(Dice_stomach[-1]))
        dsc.append(np.mean(Dice_aorta[-1]))
        dsc.append(np.mean(Dice_pancreas[-1]))
        fw.write('DSC:'+str(np.mean(dsc))+'\n')

        HD.append(hd_spleen[-1])
        HD.append(hd_right_kidney[-1])
        HD.append(hd_left_kidney[-1])
        HD.append(hd_gallbladder[-1])
        HD.append(hd_liver[-1])
        HD.append(hd_stomach[-1])
        HD.append(hd_aorta[-1])
        HD.append(hd_pancreas[-1])
        fw.write('hd:'+str(np.mean(HD))+'\n')

        nsd.append(nsd_spleen[-1])
        nsd.append(nsd_right_kidney[-1])
        nsd.append(nsd_left_kidney[-1])
        nsd.append(nsd_gallbladder[-1])
        nsd.append(nsd_liver[-1])
        nsd.append(nsd_stomach[-1])
        nsd.append(nsd_aorta[-1])
        nsd.append(nsd_pancreas[-1])
        fw.write('nsd:'+str(np.mean(nsd))+'\n')

    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    fw.write('Dice_right_kidney'+str(np.mean(Dice_right_kidney))+'\n')
    fw.write('Dice_left_kidney'+str(np.mean(Dice_left_kidney))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')

    fw.write('Mean_hd\n')
    fw.write('hd_spleen'+str(np.mean(hd_spleen))+'\n')
    fw.write('hd_right_kidney'+str(np.mean(hd_right_kidney))+'\n')
    fw.write('hd_left_kidney'+str(np.mean(hd_left_kidney))+'\n')
    fw.write('hd_gallbladder'+str(np.mean(hd_gallbladder))+'\n')
    fw.write('hd_liver'+str(np.mean(hd_liver))+'\n')
    fw.write('hd_stomach'+str(np.mean(hd_stomach))+'\n')
    fw.write('hd_aorta'+str(np.mean(hd_aorta))+'\n')
    fw.write('hd_pancreas'+str(np.mean(hd_pancreas))+'\n')

    fw.write('Mean_nsd\n')
    fw.write('nsd_spleen'+str(np.mean(nsd_spleen))+'\n')
    fw.write('nsd_right_kidney'+str(np.mean(nsd_right_kidney))+'\n')
    fw.write('nsd_left_kidney'+str(np.mean(nsd_left_kidney))+'\n')
    fw.write('nsd_gallbladder'+str(np.mean(nsd_gallbladder))+'\n')
    fw.write('nsd_liver'+str(np.mean(nsd_liver))+'\n')
    fw.write('nsd_stomach'+str(np.mean(nsd_stomach))+'\n')
    fw.write('nsd_aorta'+str(np.mean(nsd_aorta))+'\n')
    fw.write('nsd_pancreas'+str(np.mean(nsd_pancreas))+'\n')

    fw.write('*'*20+'\n')

    dsc = []
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_pancreas))
    fw.write('dsc:'+str(np.mean(dsc))+'\n')

    HD = []
    HD.append(np.mean(hd_spleen))
    HD.append(np.mean(hd_right_kidney))
    HD.append(np.mean(hd_left_kidney))
    HD.append(np.mean(hd_gallbladder))
    HD.append(np.mean(hd_liver))
    HD.append(np.mean(hd_stomach))
    HD.append(np.mean(hd_aorta))
    HD.append(np.mean(hd_pancreas))
    fw.write('hd:'+str(np.mean(HD))+'\n')

    nsd = []
    nsd.append(np.mean(nsd_spleen))
    nsd.append(np.mean(nsd_right_kidney))
    nsd.append(np.mean(nsd_left_kidney))
    nsd.append(np.mean(nsd_gallbladder))
    nsd.append(np.mean(nsd_liver))
    nsd.append(np.mean(nsd_stomach))
    nsd.append(np.mean(nsd_aorta))
    nsd.append(np.mean(nsd_pancreas))
    fw.write('nsd:'+str(np.mean(nsd))+'\n')
    print('done')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("fold", help="fold name")
    # args = parser.parse_args()
    fold = "nnformer_synapse"
    test(fold)
