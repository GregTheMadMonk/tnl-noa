#!/bin/bash

for i in $(find . -type f -name 'Makefile' | sed -r 's|/[^/]+$||' |sort |uniq)
do
	echo going into $i
	cd "$i"
	echo starting...
	make clean
	make
	make measure
	echo done measuring
	make clean
	echo going out...
	cd -
done
