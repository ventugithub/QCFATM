#!/bin/bash
for f in $( ls *.tex)
do
	#aspell -l en_US -c $f
    echo $f;
    aspell --conf="$PWD/aspell.conf" --personal="$PWD/aspell.pws" check $f
done;
	
