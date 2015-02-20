INC=-I../../include/common/ -I../../include/dvs/ -I../../include/collision/ -I../../include/kernel/ -I../../include/solver/ -I../../include/io/
LIB=-L../../include/common/ -L../../include/dvs/ -L../../include/collision/ -L../../include/kernel/ -L../../include/solver/ -L../../include/io/
VPATH=../../include/common/:../../include/dvs/:../../include/collision/:../../include/kernel/:../../include/solver/:../../include/io/
CC=g++
CFLAGS=-c -fopenmp -O3
LFLAGS=-fopenmp -O3

all: jyu_lb_isothermal

jyu_lb_isothermal: dirs lattice.o collision.o kernel.o solver.o parser.o raw_io.o main.o
		   $(CC) $(LFLAGS) lattice.o collision.o kernel.o solver.o parser.o raw_io.o main.o -o ../../bin/jyu_lb_isothermal3D.exe


dirs:
	@mkdir -p ../../bin

lattice.o: lattice.cpp
	$(CC) $(CFLAGS) $< -o $@

collision.o: lb_collision.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

kernel.o: lb_kernel.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

solver.o: lb_solver.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

parser.o: lb_input_parser.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

raw_io.o: lb_raw_io.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

main.o: jyu_lb_isothermal3D.cpp
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm *.o
