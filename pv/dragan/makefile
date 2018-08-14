CC = gcc
CFLAGS=-g -Wall -fopenmp
IDIR   = -I/usr/local/include  -I/opt/local/include  
LDIR   = -L/usr/local/lib      -L/opt/local/lib

sig:   signal.o functions.o run_camb_get_Tk.o  spline_functions.o get_sn_cov.o 
	$(CC) $(CFLAGS) -o  sig signal.o functions.o run_camb_get_Tk.o spline_functions.o get_sn_cov.o $(LDIR) -lgsl -lgslcblas -lrecipes -lm
