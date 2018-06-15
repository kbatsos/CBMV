from __future__ import division
import os
import re
import sys
import struct

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def load(fname):
  color = None
  width = None
  height = None
  scale = None
  endian = None
  
  file = open(fname)
  header = file.readline().rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')
 
  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')
 
  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian
 
  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.flipud(np.reshape(data, shape)).astype(np.float32), scale

def save(fname, image, scale=1):
  file = open(fname, 'w') 
  color = None
 
  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')
 
  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))
 
  endian = image.dtype.byteorder
 
  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale
 
  file.write('%f\n' % scale)
 
  np.flipud(image).tofile(file)

def show(img):
  imgplot = plt.imshow(img.astype(np.float32), cmap='gray');
  plt.show();
