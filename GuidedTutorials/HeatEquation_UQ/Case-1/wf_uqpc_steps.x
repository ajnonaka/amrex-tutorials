#!/bin/bash -e

#SDIR=`dirname "$0"`
export KLPC=$(pwd)/../../../../pytuq

#First-order Polynomial Chaos (PC).
## Given mean and standard deviation of each normal random parameter
echo "1 0.25 " > param_margpc.txt
echo "1 0.25" >> param_margpc.txt
echo "0.01 0.0025" >> param_margpc.txt

echo "diffusion_coeff" > pnames.txt
echo "init_amplitude" >> pnames.txt
echo "init_width" >> pnames.txt

echo "max_temp" > outnames.txt
echo "mean_temp" >> outnames.txt
echo "std_temp" >> outnames.txt
echo "total_energy" >> outnames.txt

PCTYPE="HG"
ORDER=1
NSAM=111

${KLPC}/apps/pc_prep.py marg param_margpc.txt $ORDER
#${KLPC}/apps/pc_sam.py pcf.txt $PCTYPE $NSAM

DIM=3
NTRN=$NSAM
NTST=0

./uq_pc.py -r offline_prep -c pcf.txt -x ${PCTYPE} -d $DIM -o ${ORDER} -m lsq -s rand -n $NTRN -v $NTST

cp ptrain.txt qsam.txt
#cp qsam.txt ptrain.txt

parallel --jobs 6 --keep-order --colsep ' ' \
  './main3d.gnu.ex inputs diffusion_coeff={1} init_amplitude={2} init_width={3} \
    datalog=datalog_{#}.txt \
    plot_int = -1 > /dev/null 2>&1 \
    && tail -1 datalog_{#}.txt' \
  :::: qsam.txt > ysam.txt

cp ysam.txt ytrain.txt

#${KLPC}/apps/pc_fit.py --pctype $PCTYPE --order $ORDER --xdata "qsam.txt" --ydata "ysam.txt"
./uq_pc.py -r offline_post -c pcf.txt -x ${PCTYPE} -d $DIM -o ${ORDER} -m lsq -s rand -n $NTRN -v $NTST -t 5

./plot.py sens main

./plot.py jsens
