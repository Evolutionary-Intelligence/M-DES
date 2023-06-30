# README

# .out

**.out** is the generated **logging** file when the Python script is runned for the considered optimizer, which is also named by the optimizer name.

* "DLMCMA_X_X.out" means the code started from X (X is the number of the benchmark function used in our experiments) to X (X is the number of the benchmark function).
* "DLMCMA_X_X-S*-E*.out" means that the code started from S* (* is the number of the benchmark function) but broked in E* (* is the number of the benchmark function).

Totally, we observed only 9 failures from an amount of 100 runnings (=10*10) when using the [ray](https://www.ray.io/) clustering computing software.