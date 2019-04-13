include .env

B=docker build
R=docker run
L=$(NAME)-lab:$(LAB)
V=-v `pwd`/src:/workdir -v `pwd`/data:/workdir/data

ci:
	time $B -f ci.dockerfile -t $(NAME):$(CI) .
lab:
	time $B -f ci_lab.dockerfile --build-arg CI=$(NAME):$(CI) -t $L .
sh:
	$R --rm -ti $L sh
run-lab:
	$R --rm -p 8888:80 $V $L
