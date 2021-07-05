from __future__ import absolute_import, division, print_function
from simtbx.command_line.stage_one import save_model_from_refiner
from dxtbx.model.experiment_list import ExperimentListFactory
from simtbx.diffBragg import utils
import os

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.stage_two

from libtbx.mpi4py import MPI
import time

COMM = MPI.COMM_WORLD

if COMM.rank > 0:
    import warnings
    warnings.filterwarnings("ignore")


from simtbx.diffBragg.phil import philz
from simtbx.diffBragg import ensemble_refine_launcher
from libtbx.phil import parse
from dials.util import show_mail_on_error

help_message = "stage 2 (global) diffBragg refinement"

script_phil = """
debug = False
  .type = bool
  .help = debug flag
pandas_table = None
  .type = str
  .help = path to an input pandas table (usually output by simtbx.diffBragg.predictions)
save_list = None
  .type = str
  .help = a text file with a single column specifying the opt_exp_name from pandas_table
  .help = which should be saved after the refiner exits
"""

philz = script_phil + philz
phil_scope = parse(philz)

class Script:

    def __init__(self):
        from dials.util.options import OptionParser

        self.parser = None
        if COMM.rank == 0:
            self.parser = OptionParser(
                usage="",  # stage 2 multi-shot diffBragg refinement
                sort_options=True,
                phil=phil_scope,
                read_experiments=False,
                read_reflections=False,
                check_format=False,
                epilog=help_message)
        self.parser = COMM.bcast(self.parser)

    def run(self):
        self.params = None
        if COMM.rank == 0:
            self.params, _ = self.parser.parse_args(show_diff_phil=True)
        self.params = COMM.bcast(self.params)
        if self.params.pandas_table is None:
            raise ValueError("Pandas table input required")

        refine_starttime = time.time()
        if not self.params.refiner.randomize_devices:
            self.params.simulator.device_id = COMM.rank % self.params.refiner.num_devices
        refiner = ensemble_refine_launcher.global_refiner_from_parameters(self.params)

        print("Time to refine experiment: %f" % (time.time()- refine_starttime))

        # Save models ?
        opt_exp_names = self.load_save_list()
        if opt_exp_names:
            assert self.params.refiner.io.output_dir is not None
            assert refiner.FNAMES is not None
            img_dir = os.path.join(self.params.refiner.io.output_dir, "imgs")
            if COMM.rank==0:
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
        COMM.Barrier()

        shot_ids = {fname: idx for idx,fname in refiner.FNAMES.items()}
        if COMM.rank==0:
            print(list(refiner.FNAMES.items()))
        for i_fname, fname in enumerate(opt_exp_names):
            if fname in shot_ids:
                print("Saving shot %s" % fname)
                El = ExperimentListFactory.from_json_file(fname)
                exper = El[0]
                output_dir = os.path.join(img_dir, "rank%d" % COMM.rank)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                basename = os.path.basename(fname)
                shot_name = "%s_%d.h5" %(basename, i_fname)
                imgpath = os.path.join(output_dir, shot_name)
                #if self.params.debug:
                #refiner.special_flag = 1
                save_model_from_refiner(imgpath, refiner, exper, shot_ids[fname], self.params.refiner.adu_per_photon,
                                        only_Z=True)

        #TODO save MTZ

    def load_save_list(self):
        save_list = []
        if self.params.save_list is not None:
            lines = open(self.params.save_list, "r").readlines()
            for l in lines:
                fname = l.strip()
                if not os.path.exists(fname):
                    raise IOError("Path %s from the save_list files %s does not exist!" % (l, self.params.save_list))
                save_list.append(fname)
        return save_list


if __name__ == '__main__':
    with show_mail_on_error():
        script = Script()
        script.run()
