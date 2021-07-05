#!/bin/bash -l

#SBATCH -q special    # regular or special queue
#SBATCH -N 1        # Number of nodes
#SBATCH -t 04:00:00   # wall clock time limit
#SBATCH -J dual 
#SBATCH -L SCRATCH    # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -A m1759      # allocation
#SBATCH --gres=gpu:8          # devices per node
#SBATCH -c 80         # total threads requested per node
#SBATCH -o job%j_bb2_3.out
#SBATCH -e job%j_bb2_3.err
#SBATCH --exclusive


TIMER(){ 
    export start=$(date +%s%3N);
    $@ 
    export stop=$(date +%s%3N)
    export dur=$(bc <<< "scale=2
    ($stop-$start)/1000")
    echo "COMMAND '${@}'" RUNTIME WAS $dur" seconds"
}

source ~/stable.cuda.Z.sh
export odir=bb2_3
export nprocess=753
export NPROC=32
export NPROC_PER_NODE=32
export MAX_NPROC=40
export NDEV=8
export DIFFBRAGG_USE_CUDA=1
echo -n "Begin: ";date
# first do a quick refinement to align the models 
#time srun -n $NPROC -c2 --tasks-per-node=$NPROC_PER_NODE simtbx.diffBragg.hopper $DDZ/hopper_bb.phil outdir=$odir exp_ref_spec_file=7534_exper_refl_newlams_p05.txt max_process=$nprocess niter=10 num_devices=$NDEV

# group together the pickle files and make some figures
#TIMER libtbx.python $DDZ/stg1_view_mp.py  --glob $odir --save $odir.pdf --n 1 --j 40 --symbol P6522

# next, expand the group A spots
#time srun -n$MAX_NPROC -c2  libtbx.python $DDZ/compare_spots_and_predictions.py  --pkl  ${odir}_pandas.pkl --outputExpRef ${odir}_expanded.txt --tag expanded --phil $DDZ/hopper_bb.phil

# refine the expanded group A spot models
#export odir2=${odir}_exp
#TIMER srun -n $NPROC -c2 --tasks-per-node=$NPROC_PER_NODE simtbx.diffBragg.hopper $DDZ/hopper_bb.phil outdir=$odir2 exp_ref_spec_file=${odir}_expanded.txt niter=10 num_devices=$NDEV
#
# again, group together the pickle files and make some figures
#TIMER libtbx.python $DDZ/stg1_view_mp.py  --glob $odir2 --save $odir2.pdf --n 1 --j 10 --symbol P6522

TIMER libtbx.python $DDZ/resolution_outliers.py  --input ${odir}_pandas_expanded.pkl  --output ${odir}_Zfilt_1515.pkl --threshR 1005 --threshH 1005  --reflOutdir=$odir/filtered_refls_inf --expref $odir/filtered_exp_ref_inf.txt  --save

export odir3=${odir}_ens_filtInf
TIMER srun -n$NPROC -c2 --tasks-per-node=$NPROC_PER_NODE simtbx.diffBragg.ensemble ensemble.phil outdir=$odir3 exp_ref_spec_file=${odir}/filtered_exp_ref_inf.txt best_pickle=${odir}_pandas.pkl oversample=2 maxs.G=100000 mins.G=0 sigma_r=10 temp=0.01  stepsize=0.05 fit_tilt=True fit_tilt_using_weights=False betas.G=1e5 centers.G=50 niter=0 num_devices=$NDEV  x_write_freq=25 roi.deltaQ=0.015 

#TIMER srun -n$NPROC -c2 --tasks-per-node=$NPROC_PER_NODE simtbx.diffBragg.ensemble ensemble.phil outdir=$odir3 exp_ref_spec_file=${odir}/filtered_exp_ref_1010.txt best_pickle=${odir}_pandas.pkl oversample=2 maxs.G=100000 mins.G=0 sigma_r=10 temp=0.01  stepsize=0.05 fit_tilt=True fit_tilt_using_weights=False betas.G=1e5 centers.G=50 niter=0 num_devices=$NDEV  x_write_freq=25 roi.deltaQ=0.015 

echo -n "End: ";date

