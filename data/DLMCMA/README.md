# README

# .out

**.out** is the generated **logging** file when the Python script is runned for the considered optimizer, which is also named by the optimizer name.

* "DLMCMA_X_X.out" means the code started from X (X is the number of the benchmark function used in our experiments) to X (X is the number of the benchmark function).
* "DLMCMA_X_X-S*-E*.out" means that the code started from S* (* is the number of the benchmark function) but broked in E* (* is the number of the benchmark function).

Totally, we observed only 9 failures from an amount of 100 runnings (=10*10) when using the [ray](https://www.ray.io/) clustering computing software.

#

```bash
nohup python run_experiments.py -s=1 -e=1 -o=DLMCMA >DLMCMA_1_1.out 2>&1 &
nohup python run_experiments.py -s=2 -e=2 -o=DLMCMA >DLMCMA_2_2.out 2>&1 &
nohup python run_experiments.py -s=3 -e=3 -o=DLMCMA >DLMCMA_3_3.out 2>&1 &
nohup python run_experiments.py -s=4 -e=4 -o=DLMCMA >DLMCMA_4_4.out 2>&1 &
nohup python run_experiments.py -s=5 -e=5 -o=DLMCMA >DLMCMA_5_5.out 2>&1 &
nohup python run_experiments.py -s=6 -e=6 -o=DLMCMA >DLMCMA_6_6.out 2>&1 &
nohup python run_experiments.py -s=7 -e=7 -o=DLMCMA >DLMCMA_7_7.out 2>&1 &
nohup python run_experiments.py -s=8 -e=8 -o=DLMCMA >DLMCMA_8_8.out 2>&1 &
nohup python run_experiments.py -s=9 -e=9 -o=DLMCMA >DLMCMA_9_9.out 2>&1 &
nohup python run_experiments.py -s=10 -e=10 -o=DLMCMA >DLMCMA_10_10.out 2>&1 &
```
