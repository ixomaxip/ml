include .env

B=docker build
R=docker run
L=$(NAME):$(CI)
V=-v `pwd`/src:/workdir

lab:
	time $B -f ci.dockerfile -t $L .
sh:
	$R --rm -ti $L sh
run:
	$R --rm -p 8888:80 $V $L
