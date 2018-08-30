CC = mpicc 
CFLAGS=-g -Wall -fopenmp -std=c99
IDIR   = -I/usr/local/include  -I../../nr/opt/local/include
LDIR   = -L/usr/local/lib      -L../../nr/lib

sig:   signal.o functions.o run_camb_get_Tk.o  spline_functions.o get_sn_cov.o 
	$(CC) $(CFLAGS) $(GSL) -o  sig signal.o functions.o run_camb_get_Tk.o spline_functions.o get_sn_cov.o $(LDIR) -lgsl -lgslcblas -lrecipes -lm 

.c.o:
	$(CC) -c -o $@ $(CFLAGS) $(GSL) $(IDIR) $<
clean:
	-rm -f *.o 
