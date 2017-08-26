#! /bin/bash

i=1
for file in ./png/org/*.png
	do
		cp ${file} ./png/test/$i.png
		i=`expr $i + 1`
	done
