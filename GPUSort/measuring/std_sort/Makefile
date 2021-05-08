main: main.cpp
	g++ -Wall -pedantic -std=c++14 -O3 main.cpp -o main

measure: main
	./main ../results/std_sort.csv

run: main
	./main

.PHONY: main
clean:
	rm -f main
