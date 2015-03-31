INC=-I../../include/common/ -I../../include/dvs/ -I../../include/collision/ -I../../include/kernel/ -I../../include/solver/ -I../../include/io/
LIB=-L../../include/common/ -L../../include/dvs/ -L../../include/collision/ -L../../include/kernel/ -L../../include/solver/ -L../../include/io/
VPATH=../../include/common/:../../include/dvs/:../../include/collision/:../../include/kernel/:../../include/solver/:../../include/io/
CC=g++
CFLAGS=-c -fopenmp -O3
LFLAGS=-fopenmp -O3

all: jyu_lb_isothermal

jyu_lb_isothermal: dirs node.o filter.o collision.o kernel.o solver.o parser.o raw_io.o main.o
		   $(CC) $(LFLAGS) node.o filter.o collision.o kernel.o solver.o parser.o raw_io.o main.o -o ../../bin/jyu_lb_isothermal.exe


dirs:
	@mkdir -p ../../bin

node.o: node.cpp
	$(CC) $(CFLAGS) $< -o $@

filter.o: filter.cpp
	$(CC) $(CFLAGS) $< -o $@

collision.o: collision.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

kernel.o: kernel.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

solver.o: solver.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

parser.o: input_parser.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

raw_io.o: raw_io.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

main.o: jyu_lb_isothermal.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm *.o
