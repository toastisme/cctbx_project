from __future__ import absolute_import, division, print_function
import sys
from cctbx_website.regression.exercise import exercise

def run():
  return_code = exercise(script   = "script_lbfgs_no_curvature.py",
                         tmp_path = 'tmp_files_14')
  return return_code

if __name__ == '__main__':
  sys.exit(run())
