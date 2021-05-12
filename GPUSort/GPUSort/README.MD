## Code implemented by Nguyen Xuan Thang for bachelor thesis

Needs to have CUDA and TNL installed.

* benchmark
    * folder containing benchmarking scripts
    * the main function is in ``benchmarker.cpp``
    * for each implemented algorithm, there is a folder with a benchmarker and a Makefile, to test out the algorithm, run ``make run``, to clean up ``make clean``
* src
    * folder containing the implementation of Bitonic sort, Quick sort and CDP Quick sort
    * inside each folder there is a ``sample`` folder
        * to test out the algorithm, simply run ``make run``
* tests
    * folder containing unit tests for each algorithm
    * inside each folder there is a tester and a Makefile
        * to test out the implementation, run ``make run``
    * needs gTests installed



* To install TNL, read https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/#installation
* To install CUDA, https://developer.nvidia.com/cuda-downloads
