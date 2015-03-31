VPATH=source/isothermal/

all: isothermal

isothermal:
	cd source/isothermal/; make
	cd source/isothermal/; rm *.o

