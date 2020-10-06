
# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.powder_from_spots
import logging
from iotbx.phil import parse

from scitbx.array_family import flex

from dials.util import log
from dials.util import show_mail_on_error
from dials.util.options import OptionParser
from dials.util.version import dials_version

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import pickle

logger = logging.getLogger("dials.command_line.powder_from_spots")

help_message = """
Script to synthesize a powder pattern from DIALS spotfinding output
Example usage:
cctbx.xfel.powder_from_spots all.expt all.refl

This computes d-spacings of peak maxima in a reflections file as generated by
dials.find_spots. The results are binned and plotted as a histogram. Plotting
only maxima (and thus discarding the rest of the peak profile) has a large
sharpening effect; the resulting patterns are better than those obtained by 
synchrotron powder diffraction.

The input is a single combined .refl file and the corresponding .expt file.
For small-molecule data, 10k to 20k shots are a minimum for good results.
Consider filtering the input by number of spots using the min_... and
max_reflections_per_experiment options of dials.combine_experiments. Min=3
and max=15 are a good starting point for small-molecule samples. 

An excellent reference geometry (rmsd <<1 px) is important. A current detector
metrology refined from a protein sample is probably the best approach. Try a
plot with split_detectors=True to confirm that the patterns on each panel agree.
In a data set from the MPCCD detector at SACLA we found that the Tau2 and Tau3
tilts had to be refined for each panel.
"""

phil_scope = parse(
    """
  file_path = None
    .type = str
    .multiple = True
    .help = Files to read
  n_bins = 10000
    .type = int
    .help = Number of bins in the radial average
  d_max = 20
    .type = float
  d_min = 1.4
    .type = float
  verbose = True
    .type = bool
    .help = Extra logging information
  mask = None
    .type = str
    .help = DIALS style pixel mask. Average will skip these pixels. Not \
        implemented.
  x_axis = *two_theta q resolution
    .type = choice
    .help = Units for x axis
  panel = None
    .type = int
    .help = Only use data from the specified panel
  reference_geometry = None
    .type = path
    .help = Apply this geometry before creating average. Not implemented.
  unit_cell = None
    .type = unit_cell
    .help = Show positions of miller indices from this unit_cell and space \
            group. Not implemented.
  space_group = None
    .type = space_group
    .help = Show positions of miller indices from this unit_cell and space \
            group. Not implemented.
  peak_position = *xyzobs shoebox
    .type = choice
    .help = By default, use the d-spacing of the peak maximum. Shoebox: Use the \
            coordinates of every pixel in the reflection shoebox.
  peak_weighting = *unit intensity
    .type = choice
    .help = The histogram may be intensity-weighted, but the results are \
            typically not very good.
  split_detectors = False
    .type = bool
    .help = Plot a pattern for each detector panel.
output {
  log = dials.powder_from_spots.log
    .type = str
  d_table = d_table.pkl
    .type = str
  xy_file = None
    .type = str
}
"""
)


class Script(object):
  def __init__(self):
    usage = "usage: make a powder pattern"
    self.parser = OptionParser(
        usage=usage,
        phil=phil_scope,
        epilog=help_message,
        check_format=False,
        read_reflections=True,
        read_experiments=True,
        )

  def run(self):
    params, options = self.parser.parse_args(show_diff_phil=False)

    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())


    assert len(params.input.reflections) == 1, "Please supply 1 reflections file"
    assert len(params.input.experiments) == 1, "Please supply 1 experiments file"

    # setup limits and bins
    assert params.n_bins, "Please supply n_bins for the pattern"
    n_bins = params.n_bins
    d_max, d_min = params.d_max, params.d_min
    d_inv_low, d_inv_high = 1/d_max, 1/d_min

    #sums = flex.double(params.n_bins)

    sums0 = flex.double(params.n_bins)
    sums1 = flex.double(params.n_bins)
    sums2 = flex.double(params.n_bins)
    sums3 = flex.double(params.n_bins)
    sums4 = flex.double(params.n_bins)
    sums5 = flex.double(params.n_bins)
    sums6 = flex.double(params.n_bins)
    sums7 = flex.double(params.n_bins)
    panelsums = {
        0: sums0,
        1: sums1,
        2: sums2,
        3: sums3,
        4: sums4,
        5: sums5,
        6: sums6,
        7: sums7,
        }
    d_table = []

    refls = params.input.reflections[0].data
    expts = params.input.experiments[0].data

    import random
    for i, expt in enumerate(expts):
        if random.random() < 0.01: print("experiment ", i)

        
        s0 = expt.beam.get_s0()
        sel = refls['id'] == i
        refls_sel = refls.select(sel)
        xyzobses = refls_sel['xyzobs.px.value']
        intensities = refls_sel['intensity.sum.value']
        panels = refls_sel['panel']
        shoeboxes = refls_sel['shoebox']

        for i_refl in range(len(refls_sel)):
            i_panel = panels[i_refl]
            #if i_panel not in [1,2,5,6]: continue
            panel = expt.detector[i_panel]
            sb = shoeboxes[i_refl]
            sbpixels = zip(sb.coords(), sb.values())

            
            xy = xyzobses[i_refl][0:2]
            intensity = intensities[i_refl]
            res = panel.get_resolution_at_pixel(s0, xy)
            d_table.append((res, intensity))
            if params.peak_position=="xyzobs":
                res_inv = 1/res
                i_bin = int(n_bins * (res_inv - d_inv_low) / (d_inv_high - d_inv_low))
                if i_bin < 0 or i_bin >= n_bins: continue
                panelsums[i_panel][i_bin] += intensity if params.peak_weighting=="intensity" else 1
            if params.peak_position=="shoebox":
                for (x,y,_), value in sbpixels:
                    res = panel.get_resolution_at_pixel(s0, (x,y))
                    res_inv = 1/res
                    i_bin = int(n_bins * (res_inv - d_inv_low) / (d_inv_high - d_inv_low))
                    if i_bin < 0 or i_bin >= n_bins: continue
                    panelsums[i_panel][i_bin] += value if params.peak_weighting=="intensity" else 1

                

    xvalues = np.linspace(d_inv_low, d_inv_high, n_bins)
    fig, ax = plt.subplots()
    if params.split_detectors:
        offset = max(np.array(sums1))
        for i_sums, sums in enumerate([sums1, sums2, sums5, sums6]):
            yvalues = np.array(sums)
            plt.plot(xvalues, yvalues+0.5*i_sums*offset)
    else:
        yvalues = sum([v for v in panelsums.values()])
        plt.plot(xvalues, yvalues)
    ax.get_xaxis().set_major_formatter(tick.FuncFormatter(
        lambda x, _: "{:.3f}".format(1/x)))

    if params.output.xy_file:
        with open(params.output.xy_file, 'w') as f:
            for x,y in zip(xvalues, yvalues):
                f.write("{:.6f}\t{}\n".format(1/x, y))
    plt.show()

    with open(params.output.d_table, 'wb') as f:
        pickle.dump(d_table, f)

if __name__ == "__main__":
    with show_mail_on_error():
        script = Script()
        script.run()
