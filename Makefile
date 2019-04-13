include .env

B=docker build
R=docker run

ci:
	time $B -f ci.dockerfile -t $(NAME):$(CI)
lab:
	time $B -f ci_lab.dockerfile --build-arg CI=$(NAME):$(CI) -t $(NAME)-lab:$(LAB) .
sh:
	$R --rm -ti $(NAME)-lab:$(LAB) sh
