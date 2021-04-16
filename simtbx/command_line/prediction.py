from __future__ import absolute_import, division, print_function

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.prediction

import glob
import time
import os
try:
    import pandas
except ImportError:
    print("Pandas is required. Install using 'libtbx.python -m pip install pandas'")
    exit()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from simtbx.diffBragg import utils
from libtbx.phil import parse


help_message = "predictions using diffBragg refinement results"

script_phil = """
refl_list = None
  .type = str
  .help = refl file list
exper_list = None
  .type = str
  .help = expt file list
panda_list = None
  .type = str
  .help = pandas file list
spectra_list = None
  .type = str
  .help = .lam spectra file (precognition) list
stage_one_folder = None
  .type = str
  .multiple = True
  .help = path(s) to stage_one output folders
oversample_override = None
  .type = int
  .help = force the nanoBragg oversample property to be this value (pixel oversample rate)
default_F = 1e3
  .type = float
  .help = if not supplying an MTZ, this is the default structure factor amplitude
Ncells_abc_override = None
  .type = ints(size=3)
  .help = force the nanoBragg Ncells_abc property to be this 3-tuple (Na,Nb,Nc mosaic block size in unit cells)
pink_stride_override = None
  .type = int
  .help = if this is N, then take every Nth channel in the model spectrum
  .help = and therefore make the simulation faster
cuda = False
  .type = bool
  .help = whether to use cuda
omp = False
  .type = bool
  .help = try to use openMP
d_max = 999
  .type = float
  .help = maximum resolution
d_min = 1.4
  .type = float
  .help = minimum resolution
output_img = None
  .type = str
  .help = a name specifying an output image prefix for writing the model
outfile = None
  .type = str
  .help = output reflection file for indexed refls
prediction_tag = None
  .type = str
  .help = tag for predicted reflections 
tolerance = 1
  .type = float
  .help = indexing toleraance for assigning indices to the modeled spots
thresh = 1
  .type = float
  .help = threshold in photons for a modeled pixel to be flagged as part of a Bragg spot
Qdist_cutoff = 0.003
  .type = float
  .help = minimum distance in recip space for a strong spot to a modeled spot
  .help = in order that the strong spot be indexed
ngpu = 1
  .type = int
  .help = number of GPUs to use
njobs = 1
  .type = int
  .help = number of jbs to use, each job will use a randomly assigned gpu, up to ngpu
max_process = None
  .type = int
  .help = maximum number of imgs to predict
pandas_outfile = None
  .type = str
  .help = output file name (this file is suitable input to stage_two refinement, e.g. the pandas_table parameter)
mtz_name = None
  .type = str
  .help = path to mtz used in prediction
mtz_col = None
  .type = str
  .help = mtz col name
symbol_override = None
  .type = str
  .help = space group lookup symbol if not supplying mtz. If None, comes from crystal model
"""

phil_scope = parse(script_phil)

class Script:

    def __init__(self):
        from dials.util.options import OptionParser

        self.parser = OptionParser(
            usage="",
            sort_options=True,
            phil=phil_scope,
            read_experiments=False,
            read_reflections=False,
            check_format=False,
            epilog=help_message)

    def run(self):
        self.params, _ = self.parser.parse_args(show_diff_phil=True)
        if self.params.omp and self.params.cuda:
            raise ValueError("omp and cuda cannot be used simultaneously")

        if self.params.stage_one_folder is None:
            explist = self.load_filelist(self.params.exper_list)
            reflist = [None] * len(explist)
            if self.params.refl_list is not None:
                reflist = self.load_filelist(self.params.refl_list)
            speclist = [None]*len(explist)
            if self.params.spectra_list is not None:
                speclist = self.load_filelist(self.params.spectra_list)
            pandalist = self.load_filelist(self.params.panda_list)
            assert (len(reflist) == len(speclist) == len(explist))
            input_iterator = zip(explist, reflist, speclist, pandalist)
            auto_parsed_stage_one_folder = False
            NUM_EXPER = len(pandalist)
        else:
            panda_names = self._list_panda_files()
            NUM_EXPER = len(panda_names)
            def _input_iterator():
                for panda_name in panda_names:
                    yield None, None, None, panda_name
            input_iterator = _input_iterator()
            auto_parsed_stage_one_folder = True

        processed_frames = []
        NUM_PROCESS = NUM_EXPER
        if self.params.max_process is not None:
            NUM_PROCESS = self.params.max_process
        for i_exp, (exper_file, strong_file, spec_file, panda_file) in enumerate(input_iterator):
            # NOTE panda_file can be a dataframe or a string
            if i_exp % COMM.size != COMM.rank:
                continue

            if self.params.max_process is not None and i_exp >= self.params.max_process:
                break

            panda_frame = None
            if auto_parsed_stage_one_folder:
                panda_frame = pandas.read_pickle(panda_file)
                # Drop the high-memory consuming columns if they are present:
                cols = list(panda_frame)
                if "bgplanes" in cols:
                    panda_frame.drop(["bgplanes"], axis=1, inplace=True)
                if "bgplanes_xpos" in cols:
                    panda_frame.drop(["bgplanes_xpos"], axis=1, inplace=True)
                # fix log(x<=0)
                panda_cols = list(panda_frame)
                if "opt_exp_name" not in panda_cols:
                    print("WARNING: panda frame %s doesnt contain experiment path" % panda_file)
                    continue
                exper_file = panda_frame.opt_exp_name.values[0]
                if not os.path.exists(exper_file):
                    print("WARNING: path to experiment %s does not exist" % exper_file)
                    continue

            print("<><><><><><><><><><><><><><><><><><>")
            print("\tRank %d : iter %d / %d" % (COMM.rank, i_exp+1, NUM_PROCESS))
            print("<><><><><><><><><><><><><><><><><><>")

            El = ExperimentListFactory.from_json_file(exper_file, check_format=False)
            exper = El[0]

            # optional to read in strong reflection tables, to see which strong reflections are indexed by the new model
            if strong_file is not None:
                strong = flex.reflection_table.from_file(strong_file)
            else:
                strong = None

            tstart = time.time()
            dev_id = COMM.rank % self.params.ngpu
            output_img = self.params.output_img
            if output_img is not None:
                output_img = self.params.output_img + "_%d.h5" % i_exp
            if not auto_parsed_stage_one_folder:
                model_imgs = utils.spots_from_pandas_and_experiment(exper, panda_file,
                    spectrum_file=spec_file,
                    cuda=self.params.cuda, d_max=self.params.d_max, d_min=self.params.d_min,
                    output_img=output_img,
                    njobs=self.params.njobs, device_Id=dev_id, as_numpy_array=True, omp=self.params.omp)
            else:
                model_imgs = utils.spots_from_pandas(panda_frame,
                    mtz_file=self.params.mtz_name,
                    mtz_col=self.params.mtz_col,
                    oversample_override=self.params.oversample_override,
                    Ncells_abc_override=self.params.Ncells_abc_override,
                    pink_stride_override=self.params.pink_stride_override,
                    cuda=self.params.cuda, d_max=self.params.d_max, d_min=self.params.d_min,
                    output_img=output_img,
                    njobs=self.params.njobs, device_Id=dev_id, omp=self.params.omp, norm_by_spectrum=True,
                    symbol_override=self.params.symbol_override, defaultF=self.params.default_F)

            # if strong is None, this will just return all the predictions
            # else it returns the strong reflections that are indexed by the prediction model
            Rindexed = utils.indexed_from_model(strong, model_imgs, exper, tolerance=self.params.tolerance,
                                                thresh=self.params.thresh, Qdist_cutoff=self.params.Qdist_cutoff)
            Rindexed['id'] = flex.int(len(Rindexed), 0)
            if strong is not None:
                Rindexed = utils.remove_multiple_indexed(Rindexed)
                print("%d / %d are indexed!" % (len(Rindexed), len(strong)))

            # TODO 1/2 pix ?

            prediction_outfile = os.path.splitext(exper_file)[0] + "_%s.refl"
            tag = "diffBragg_prediction"
            if self.params.prediction_tag is not None:
                tag = self.params.prediction_tag
            prediction_outfile = prediction_outfile % tag
            Rindexed.as_file(prediction_outfile)

            tdone = time.time() - tstart
            print("Done, saved indexed refls to file %s (took %.4f sec)" % (prediction_outfile, tdone))

            if auto_parsed_stage_one_folder:
                panda_frame["predictions"] = os.path.abspath(prediction_outfile)
                processed_frames.append(panda_frame)

        if auto_parsed_stage_one_folder:
            processed_frames = COMM.reduce(processed_frames)
            if COMM.rank == 0:
                if self.params.pandas_outfile is None:
                    master_outfile = os.path.splitext(self.params.pandas_table)[0] + "_predict.pkl"
                else:
                    master_outfile = self.params.pandas_outfile
                master_frame = pandas.concat(processed_frames)
                master_frame.to_pickle(master_outfile)
                print("Saved predictions dataframe (this is ready for stage_two!): %s" % master_outfile)

    @staticmethod
    def load_filelist(fname):
        lines = [l.strip() for l in open(fname, "r").readlines()]
        for l in lines:
            if len(l.split()) > 1:
                raise RuntimeError("Input file %s is weird, needs single file path per line" % fname)
            if not os.path.exists(l):
                raise RuntimeError("fpath %s does not exist (from file list in %s)" % (l, fname))
        return lines

    def _list_panda_files(self):
        if COMM.rank == 0:
            all_panda_names = []
            for stage_one_path in self.params.stage_one_folder:
                if not os.path.exists(stage_one_path):
                    print("Path does not exist: %s" % stage_one_path)
                    continue
                panda_glob = os.path.join(stage_one_path, "pandas", "rank*", "*pkl")
                panda_names = glob.glob(panda_glob)
                print("Found %d pandas files in folder %s" % (len(panda_names), stage_one_path))
                all_panda_names += panda_names
        else:
            all_panda_names = None
        all_panda_names = COMM.bcast(all_panda_names)
        return all_panda_names


if __name__ == '__main__':
    from dials.util import show_mail_on_error
    with show_mail_on_error():
        script = Script()
        script.run()
