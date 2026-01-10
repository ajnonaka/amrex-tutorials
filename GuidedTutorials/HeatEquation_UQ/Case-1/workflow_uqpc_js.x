#!/bin/bash -e

# location of pytuq
export KLPC=$(pwd)/../../../../pytuq

#First-order Polynomial Chaos (PC).
## Given mean and standard deviation of each normal random parameter
# inputs are diffusion_coeff, init_amplitude, init_variance
echo "1 0.25 " > param_margpc.txt
echo "1 0.25" >> param_margpc.txt
echo "0.01 0.0025" >> param_margpc.txt

PCTYPE="HG"
ORDER=1
NSAM=20

# generate pcf.txt (re-ordering of parame_margpc.txt)
${KLPC}/apps/pc_prep.py marg param_margpc.txt $ORDER

# generate qsam.txt (random numbers drawn from polynomial chaos basis) and
# generate psam.txt (the randomly varying inputs parameters)
${KLPC}/apps/pc_sam.py pcf.txt $PCTYPE $NSAM

# run all the jobs with psam.txt as the parameters, and log the final outputs in ysam.txt
# outputs to ysam.txt are max_temp, mean_temp, std_temp, total_energy
parallel --jobs 4 --keep-order --colsep ' ' \
  './main3d.gnu.ex inputs diffusion_coeff={1} init_amplitude={2} init_variance={3} \
    datalog=datalog_{#}.txt \
    > /dev/null 2>&1 \
    && tail -1 datalog_{#}.txt' \
  :::: psam.txt > ysam.txt

${KLPC}/apps/pc_fit.py --pctype $PCTYPE --order $ORDER --xdata "psam.txt" --ydata "ysam.txt"
