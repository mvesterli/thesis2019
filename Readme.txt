This repository contains the code necessary to reproduce the results in my thesis.
To run the experiments, the PUFFINN library should be built via CMake.
The resulting library is then copied into the root of the ann_benchmarks folder.

An experiment can be run by executing the following script in ann_benchmarks.

python3 run.py --definitions graph-algos.yaml --algorithm plain-lsh --dataset glove-100-angular -k 10 --local.
The algorithm can be replaced by any algorithm that has been definied in graph-algos.yaml.
The dataset can be any dataset that is supported by ann-benchmarks or is located in the data folder. Note that some of the names do not match the report.

To run the algorithms KGraph, EFANNA and RKNNG, the local flag should be removed, at which point it is run through docker if it is running. The algorithm also needs to be installed via the install.py script.

To visualize the results, the following command can be used.
python3 create_website.py --scatter --plottype recall/buildtime --outputdir web 

The implemented k-NN graph algorithms are located in puffinn/include/graph.
Example usage from C++ can be seen in puffinn/examples/graph.cpp

