#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

# NOTE sbatch choices are altered here to make gnu parallel work with the ones in https://github.com/WeiqunZhang/amrex-scaling/blob/824551bc0189f374317bf8602bb81799deb970a2/fft/perlmutter/2025-02-06/run-16.sh
# NOTE this example is only meant for one node

#SDIR=`dirname "$0"`
export KLPC=$(pwd)/../../../../pytuq

#First-order Polynomial Chaos (PC).
## Given mean and standard deviation of each normal random parameter
echo "1 0.25 " > param_margpc.txt
echo "1 0.25" >> param_margpc.txt
echo "0.01 0.0025" >> param_margpc.txt

echo "diffusion_coeff" > pnames.txt
echo "init_amplitude" >> pnames.txt
echo "init_variance" >> pnames.txt

echo "max_temp" > outnames.txt
echo "mean_temp" >> outnames.txt
echo "std_temp" >> outnames.txt
echo "total_energy" >> outnames.txt

PCTYPE="HG"
ORDER=1
NSAM=111

${KLPC}/apps/pc_prep.py marg param_margpc.txt $ORDER
${KLPC}/apps/pc_sam.py pcf.txt $PCTYPE $NSAM

srun parallel --jobs 4 --keep-order --colsep ' ' \
  './main3d.gnu.ex inputs diffusion_coeff={1} init_amplitude={2} init_variance={3} \
    datalog=datalog_{#}.txt \
    plot_int = -1 > /dev/null 2>&1 \
    && tail -1 datalog_{#}.txt' \
  :::: qsam.txt > ysam.txt

${KLPC}/apps/pc_fit.py --pctype $PCTYPE --order $ORDER --xdata "qsam.txt" --ydata "ysam.txt"
