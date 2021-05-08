## measuring folder

* *.ipynb are python jupyter notebook used to process measured data
* ``script.sh`` is a bash scrip that will start all measurements and save the results into ``results`` folder
* ``results`` is a folder to store all .csv files generated after measurement
* each of the folder has a Makefile to start measuring
    * to measure an algorithm manually, go into the folder, call ``make`` and execute the binary
    * ``./a.out`` will print the results on the standard output
    * ``./a.out ../results/my_results.csv`` will save the time measured into the given file location