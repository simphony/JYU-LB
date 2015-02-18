VPATH=source/isothermal3D/

all: isothermal3D

isothermal3D:
	cd source/isothermal3D/; make
	cd source/isothermal3D/; rm *.o

