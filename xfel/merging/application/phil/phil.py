from __future__ import absolute_import, division, print_function

from iotbx.phil import parse

help_message = '''
Redesign script for merging xfel data
'''

dispatch_phil = """
dispatch {
  step_list = None
    .type = strings
    .help = List of steps to use. None means use the full set of steps to merge.
}
"""

input_phil = """
input {
  keep_imagesets = False
    .type = bool
    .help = If True, keep imagesets attached to experiments
  path = None
    .type = str
    .multiple = True
    .help = paths are validated as a glob, directory or file.
    .help = however, validation is delayed until data are assigned to parallel ranks.
    .help = integrated experiments (.expt) and reflection tables (.refl) must both be
    .help = present as matching files.  Only one need be explicitly specified.
  reflections_suffix = _integrated.refl
    .type = str
    .help = Find file names with this suffix for reflections
  experiments_suffix = _integrated.expt
    .type = str
    .help = Find file names with this suffix for experiments

  parallel_file_load {
    method = *uniform node_memory
      .type = choice
      .help = uniform: distribute input experiments/reflections files uniformly over all available ranks
      .help = node_memory: distribute input experiments/reflections files over the nodes such that the node memory limit is not exceeded.
      .help = Within each node distribute the input files uniformly over all ranks of that node.
    node_memory {
      architecture = "Cori KNL"
        .type = str
        .help = node architecture name. Currently not used.
      limit = 90.0
        .type = float
        .help = node memory limit, GB. On Cori KNL each node has 96 GB of memory, but we use 6 GB as a cushion, so the default value is 90 GB.
      pickle_to_memory = 3.5
        .type = float
        .help = an empirical coefficient to convert pickle file size to anticipated run-time process memory required to load a file of that size
    }
    ranks_per_node = 68
        .type = int
        .help = number of MPI ranks per node
    balance = *global per_node
      .type = choice
      .multiple = False
      .help = Balance the input file load by distributing experiments uniformly over all available ranks (global) or over the ranks on each node (per_node)
      .help = The idea behind the "per_node" method is that it doesn't require MPI communications across nodes. But if the input file load varies strongly
      .help = between the nodes, "global" is a much better option.
    balance_mpi_alltoall_slices = 1
      .type = int
      .expert_level = 2
      .help = memory reduction factor for MPI alltoall.
      .help = Use mpi_alltoall_slices > 1, when available RAM is insufficient for doing MPI alltoall on all data at once.
      .help = The data will then be split into mpi_alltoall_slices parts and, correspondingly, alltoall will be performed in mpi_alltoall_slices iterations.
    reset_experiment_id_column = False
      .type = bool
      .expert_level = 3
  }
}

mp {
  method = *mpi
    .type = choice
    .help = Muliprocessing method (only mpi at present)
}
"""

tdata_phil = """
tdata{
  output_path = None
    .type = path
    .help = If output_path is not None, the tdata worker writes out a list of unit cells to a file.
    .help = Generally speaking the program should then stop.  The tdata worker is not active by default, so it is necessary to have
    .help = the following phil configuration: dispatch.step_list=input,tdata.
    .help = The output_path assumes the *.tdata filename extension will be appended.
    .help = More information about using this option is given in the source code, xfel/merging/application/tdata/README.md
}
"""

filter_phil = """
filter
  .help = The filter section defines criteria to accept or reject whole experiments
  .help = or to modify the entire experiment by a reindexing operator
  .help = refer to the select section for filtering of individual reflections
  {
  algorithm = n_obs a_list reindex resolution unit_cell report
    .type = choice
    .multiple = True
  n_obs {
    min = 15
      .type = int
      .help = Minimum number of observations for subsequent processing
  }
  a_list
    .help = a_list is a text file containing a list of acceptable experiments
    .help = for example, those not misindexed, wrong type, or otherwise rejected as determined separately
    .help = suggested use, string matching, can include timestamp matching, directory name, etc
    {
    file = None
      .type = path
      .multiple = True
    operation = *select deselect
      .type = choice
      .multiple = True
      .help = supposedly have same number of files and operations. Different lists can be provided for select and deselect
  }
  reindex {
    data_reindex_op = h,k,l
      .type = str
      .help = Reindex, e.g. to change C-axis of an orthorhombic cell to align Bravais lattice from indexing with actual space group
    reverse_lookup = None
      .type = str
      .help = filename, pickle format, generated by the cxi.brehm_diederichs program.  Contains a
      .help = (key,value) dictionary where key is the filename of the integrated data pickle file (supplied
      .help = with the data phil parameter and value is the h,k,l reindexing operator that resolves the
      .help = indexing ambiguity.
    sampling_number_of_lattices = 1000
      .type = int
      .help = Number of lattices to be gathered from all ranks to run the brehm-diederichs procedure
  }
  resolution {
    d_min = None
      .type = float
      .help = Reject the experiment unless some reflections extend beyond this resolution limit
    model_or_image = model image
      .type = choice
      .help = Calculate resolution either using the scaling model unit cell or from the image itself
  }
  unit_cell
    .help = Various algorithms to restrict unit cell and space group
    {
    algorithm = range *value cluster
      .type = choice
    value
      .help = Discard lattices that are not close to the given target.
      .help = If the target is left as Auto, use the scaling model
      .help = (derived from either PDB file cryst1 record or MTZ header)
      {
      target_unit_cell = Auto
        .type = unit_cell
      relative_length_tolerance = 0.1
        .type = float
        .help = Fractional change in unit cell dimensions allowed (versus target cell).
      absolute_angle_tolerance = 2.
        .type = float
      target_space_group = Auto
        .type = space_group
      }
    cluster
      .help = CLUSTER implies an implementation (standalone program or fork?) where all the
      .help = unit cells are brought together prior to any postrefinement or merging,
      .help = and analyzed in a global sense to identify the isoforms.
      .help = the output of this program could potentially form the a_list for a subsequent
      .help = run where the pre-selected events are postrefined and merged.
      {
      algorithm = rodgriguez_laio dbscan *covariance
        .type = choice
      covariance
        .help = Read a pickle file containing the previously determined clusters,
        .help = represented by estimated covariance models for unit cell parameters.
        {
        file = None
          .type = path
        component = 0
          .type = int(value_min=0)
        mahalanobis = 4.0
          .type = float(value_min=0)
          .help = Is essentially the standard deviation cutoff. Given that the unit cells
          .help = are estimated to be distributed by a multivariate Gaussian, this is the
          .help = maximum deviation (in sigmas) from the central value that is allowable for the
          .help = unit cell to be considered part of the cluster.
        }
      isoform = None
        .type=str
        .help = unknown at present. if there is more than one cluster, such as in PSII,
        .help = perhaps the program should write separate a_lists.
        .help = Alternatively identify a particular isoform to carry forward for merging.
      }
  }
  outlier {
    min_corr = 0.1
      .type = float
      .help = Correlation cutoff for rejecting individual experiments by comparing observed intensities to the model.
      .help = This filter is not applied if scaling.model==None. No experiments are rejected with min_corr=-1.
      .help = This either keeps or rejects the whole experiment.
    assmann_diederichs {}
  }
}
"""

modify_phil = """
modify
  .help = The MODIFY section defines operations on the integrated intensities
  {
  algorithm = *polarization
    .type = choice
    .multiple = True
  reindex_to_reference
    .help = An algorithm to match input experiments against a reference model to
    .help = break an indexing ambiguity
    {
    dataframe = None
      .type = path
      .help = if not None, save a list of which experiments were reindexed (requires pandas)
      .help = and plot a histogram of correlation coefficients (matplotlib)
    }
  cosym
    .help = Implement the ideas of Gildea and Winter doi:10.1107/S2059798318002978
    .help = to determine Laue symmetry from individual symops
    {
    include scope dials.command_line.cosym.phil_scope
    dataframe = None
      .type = path
      .help = if not None, save a list of which experiments were reindexed (requires pandas)
      .help = and plot a histogram of correlation coefficients (matplotlib)
    anchor = False
      .type = bool
      .help = Once the patterns are mutually aligned with the Gildea/Winter/Brehm/Diederichs methodology
      .help = flip the whole set so that it is aligned with a reference model.  For simplicity, the
      .help = reference model from scaling.model is used.  It should be emphasized that the scaling.model
      .help = is only used to choose the overall alignment, which may be chosen arbitrarily, it does not
      .help = bias the mutual alignment of the experimental diffraction patterns.
    plot
      {
      do_plot = True
        .type = bool
        .help = Generate embedding plots to assess quality of modify_cosym reindexing.
      n_max = 1
        .type = int
        .help = If shots were divided into tranches for alignment, generate embedding plots for
        .help = the first n_max tranches.
      interactive = False
        .type = bool
        .help = Open embedding plot in Matplotlib window instead of writing a file.
      format = *png pdf
        .type = choice
        .multiple = False
      filename = cosym_embedding
        .type = str
      }
    }
}
"""

select_phil = """
select
  .help = The select section accepts or rejects specified reflections
  .help = refer to the filter section for filtering of whole experiments
  {
  algorithm = panel cspad_sensor significance_filter
    .type = choice
    .multiple = True
  cspad_sensor {
    number = None
      .type = int(value_min=0, value_max=31)
      .multiple = True
      .help = Index in the range(32) specifying sensor on the CSPAD to deselect from merging, for the purpose
      .help = of testing whether an individual sensor is poorly calibrated.
    operation = *deselect select
      .type = choice
      .multiple = True
  }
  significance_filter
    .help = If listed as an algorithm, apply a sigma cutoff (on unmerged data) to limit
    .help = the resolution from each diffraction pattern.
    .help = Implement an alternative filter for fuller-kapton geometry
    {
    n_bins = 12
      .type = int (value_min=2)
      .help = Initial target number of resolution bins for sigma cutoff calculation
    min_ct = 10
      .type = int
      .help = Decrease number of resolution bins to require mean bin population >= min_ct
    max_ct = 50
      .type = int
      .help = Increase number of resolution bins to require mean bin population <= max_ct
    sigma = 0.5
      .type = float
      .help = Remove highest resolution bins such that all accepted bins have <I/sigma> >= sigma
    }
}
"""

scaling_phil = """
scaling {
  model = None
    .type = str
    .help = PDB filename containing atomic coordinates & isomorphous cryst1 record
    .help = or MTZ filename from a previous cycle. If MTZ, specify mtz.mtz_column_F.
  unit_cell = None
    .type = unit_cell
    .help = Unit cell to be used during scaling and merging. Used if model is not provided
    .help = (e.g. mark1).
  space_group = None
    .type = space_group
    .help = Space group to be used during scaling and merging. Used if model is not provided
    .help = (e.g. mark1).
  model_reindex_op = h,k,l
    .type = str
    .help = Kludge for cases with an indexing ambiguity, need to be able to adjust scaling model
  resolution_scalar = 0.969
    .type = float
    .help = Accommodates a few more miller indices at the high resolution limit to account for
    .help = unit cell variation in the sample. merging.d_min is multiplied by resolution_scalar
    .help = when computing which reflections are within the resolution limit.
  mtz {
    mtz_column_F = fobs
      .type = str
      .help = scaling reference column name containing reference structure factors. Can be
      .help = intensities or amplitudes
    minimum_common_hkls = -1
      .type = int
      .help = minimum required number of common hkls between mtz reference and data
      .help = used to validate mtz-based model. No validation with -1.
  }
  pdb {
    include_bulk_solvent = True
      .type = bool
      .help = Whether to simulate bulk solvent
    k_sol = 0.35
      .type = float
      .help = If model is taken from coordinates, use k_sol for the bulk solvent scale factor
      .help = default is approximate mean value in PDB (according to Pavel)
    b_sol = 46.00
      .type = float
      .help = If model is taken from coordinates, use b_sol for bulk solvent B-factor
      .help = default is approximate mean value in PDB (according to Pavel)
  }
  algorithm = *mark0 mark1
    .type = choice
    .help = "mark0: original per-image scaling by reference to isomorphous PDB model"
    .help = "mark1: no scaling, just averaging (i.e. Monte Carlo
             algorithm).  Individual image scale factors are set to 1."
}
"""

postrefinement_phil = """
postrefinement {
  enable = False
    .type = bool
    .help = enable the preliminary postrefinement algorithm (monochromatic)
    .expert_level = 3
  algorithm = *rs rs2 rs_hybrid eta_deff
    .type = choice
    .help = rs only, eta_deff protocol 7
    .expert_level = 3
  rs {
    fix = thetax thetay *RS G BFACTOR
      .type = choice(multi=True)
      .help = Which parameters to fix during postrefinement
  }
  rs2
    .help = Reimplement postrefinement with the following (Oct 2016):
    .help = Refinement engine now work on analytical derivatives instead of finite differences
    .help = Better convergence using "traditional convergence test"
    {}
  rs_hybrid
    .help = More aggressive postrefinement with the following (Oct 2016):
    .help = One round of 'rs2' using LBFGS minimizer as above to refine G,B,rotx,roty
    .help = Gentle weighting rather than unit weighting for the postrefinement target
    .help = Second round of LevMar adding an Rs refinement parameter
    .help = Option of weighting the merged terms by partiality
    {
    partiality_threshold = 0.2
      .type = float ( value_min = 0.01 )
      .help = throw out observations below this value. Hard coded as 0.2 for rs2, allow value for hybrid
      .help = must enforce minimum positive value because partiality appears in the denominator
    }
  target_weighting = *unit variance gentle extreme
    .type = choice
    .help = weights for the residuals in the postrefinement target (for rs2 or rs_hybrid)
    .help = Unit: each residual weighted by 1.0
    .help = Variance: weighted by 1/sigma**2.  Doesn't seem right, constructive feedback invited
    .help = Gentle: weighted by |I|/sigma**2.  Seems like best option
    .help = Extreme: weighted by (I/sigma)**2.  Also seems right, but severely downweights weak refl
  merge_weighting = *variance
    .type = choice
    .help = assumed that individual reflections are weighted by the counting variance
  merge_partiality_exponent = 0
    .type = float
    .help = additionally weight each measurement by partiality**exp when merging
    .help = 0 is no weighting, 1 is partiality weighting, 2 is weighting by partiality-squared
  lineshape = *lorentzian gaussian
    .type = choice
    .help = Soft sphere RLP modeled with Lorentzian radial profile as in prime
    .help = or Gaussian radial profile. (for rs2 or rs_hybrid)
  show_trumpet_plot = False
    .type = bool
    .help = each-image trumpet plot showing before-after plot. Spot color warmth indicates I/sigma
    .help = Spot radius for lower plot reflects partiality. Only implemented for rs_hybrid
}
"""

merging_phil = """
merging {
  minimum_multiplicity = 2
    .type = int(value_min=2)
    .help = If defined, merged structure factors not produced for the Miller indices below this threshold.
  error {
    model = ha14 ev11 errors_from_sample_residuals
      .type = choice
      .multiple = False
      .help = ha14, formerly sdfac_auto, apply sdfac to each-image data assuming negative
      .help = intensities are normally distributed noise
      .help = errors_from_sample_residuals, use the distribution of intensities in a given miller index
      .help = to compute the error for each merged reflection
    ev11
      .help = formerly sdfac_refine, correct merged sigmas refining sdfac, sdb and sdadd as Evans 2011.
      {
      random_seed = None
        .help = Random seed. May be int or None. Only used for the simplex minimizer
        .type = int
        .expert_level = 1
      minimizer = *lbfgs LevMar
        .type = choice
        .help = Which minimizer to use while refining the Sdfac terms
      refine_propagated_errors = False
        .type = bool
        .help = If True then during sdfac refinement, also \
                refine the estimated error used for error propagation.
      show_finite_differences = False
        .type = bool
        .help = If True and minimizer is lbfgs, show the finite vs. analytical differences
      plot_refinement_steps = False
        .type = bool
        .help = If True, plot refinement steps during refinement.
    }
  }
  plot_single_index_histograms = False
    .type = bool
  set_average_unit_cell = True
    .type = bool
    .help = Output file adopts the unit cell of the data rather than of the reference model.
    .help = How is it determined?  Not a simple average, use a cluster-driven method for
    .help = deriving the best unit cell value.
  d_min = None
    .type = float
    .help = limiting resolution for scaling and merging
  d_max = None
    .type = float
    .help = limiting resolution for scaling and merging.  Implementation currently affects only the CCiso cal
  merge_anomalous = False
    .type = bool
    .help = Merge anomalous contributors
}
"""

output_phil = """
output {
  prefix = iobs
    .type = str
    .help = Prefix for all output file names
  title = None
    .type = str
    .help = Title for run - will appear in MTZ file header
  output_dir = .
    .type = str
    .help = output file directory
  tmp_dir = None
    .type = str
    .help = temporary file directory
  do_timing = False
    .type = bool
    .help = When True, calculate and log elapsed time for execution steps
  log_level = 1
    .type = int
    .help = determines how much information to log. Level 0 means: log all, while a non-zero level reduces the logging amount.
  save_experiments_and_reflections = False
    .type = bool
    .help = If True, dump the final set of experiments and reflections from the last worker
}
"""

statistics_phil = """
statistics {
  n_bins = 10
    .type = int(value_min=1)
    .help = Number of resolution bins in statistics table
  cc1_2 {
    hash_filenames = False
      .type = bool
      .help = For CC1/2, instead of using odd/even filenames to split images into two sets,
      .help = hash the filename using md5 and split the images using odd/even hashes.
  }
  cciso {
    mtz_file = None
      .type = str
      .help = for Riso/ CCiso, the reference structure factors, must have data type F
      .help = a fake file is written out to this file name if model is None
    mtz_column_F = fobs
      .type = str
      .help = for Riso/ CCiso, the column name containing reference structure factors
  }
  predictions_to_edge {
    apply = False
      .type = bool
      .help = If True and key 'indices_to_edge' not found in integration pickles, predictions
      .help = will be made to the edge of the detector based on current unit cell, orientation,
      .help = and mosaicity.
    image = None
      .type = path
      .help = Path to an example image from which to extract active areas and pixel size.
    detector_phil = None
      .type = path
      .help = Path to the detector version phil file used to generate the selected data.
  }
  report_ML = True
    .type = bool
    .help = Report statistics on per-frame attributes modeled by max-likelihood fit (expert only).
}
"""

group_phil = """
parallel {
  a2a = 1
    .type = int
    .expert_level = 2
    .help = memory reduction factor for MPI alltoall.
    .help = Use a2a > 1, when available RAM is insufficient for doing MPI alltoall on all data at once.
    .help = The data will be split into a2a parts and, correspondingly, alltoall will be performed in a2a iterations.
}
"""

publish_phil = """
publish {
  include scope xfel.command_line.upload_mtz.phil_scope
}
"""

lunus_phil = """
lunus {
  deck_file = None
    .type = path
}
"""

# A place to override any defaults included from elsewhere
program_defaults_phil_str = """
modify.cosym.use_curvatures=False
"""

master_phil = dispatch_phil + input_phil + tdata_phil + filter_phil + modify_phil + \
              select_phil + scaling_phil + postrefinement_phil + merging_phil + \
              output_phil + statistics_phil + group_phil + lunus_phil + publish_phil
phil_scope = parse(master_phil, process_includes = True)
phil_scope = phil_scope.fetch(parse(program_defaults_phil_str))

class Script(object):
  '''A class for running the script.'''

  def __init__(self):
    # The script usage
    import libtbx.load_env
    self.usage = "usage: %s [options] [param.phil] " % libtbx.env.dispatcher_name
    self.parser = None

  def initialize(self):
    '''Initialise the script.'''
    from dials.util.options import OptionParser
    # Create the parser
    self.parser = OptionParser(
      usage=self.usage,
      phil=phil_scope,
      epilog=help_message)
    self.parser.add_option(
        '--plots',
        action='store_true',
        default=False,
        dest='show_plots',
        help='Show some plots.')

    # Parse the command line. quick_parse is required for MPI compatibility
    params, options = self.parser.parse_args(show_diff_phil=True,quick_parse=True)
    self.params = params
    self.options = options

  def validate(self):
    from xfel.merging.application.validation.application import application
    application(self.params)

  def modify(self, experiments, reflections):
    return experiments, reflections #nop

  def run(self):
    print('''Initializing and validating phil...''')

    self.initialize()
    self.validate()

    # do other stuff
    return

if __name__ == '__main__':
  script = Script()
  result = script.run()
  print ("OK")
