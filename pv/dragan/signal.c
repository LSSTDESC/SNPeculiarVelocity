#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include<omp.h>
#include<mpi.h>
#include"declarations.h"
#include"recipes/nrutil.h"
#include"recipes/nr.h"
#include <time.h>
#include <string.h>

struct Orderer {
   int  index;
   int  comparator;
}; 

int cmpfunc (const void * a, const void * b) {
    struct Orderer *a1 = (struct Orderer *)a;
    struct Orderer *a2 = (struct Orderer*)b;
    return ( a1->comparator - a2->comparator );
}

int main(int argc, char *argv[])
{

    char* fileroot;
    int startindex;
    if (argc !=3) {
        printf("Specify file root\n");
        exit(1);
    }
    fileroot = argv[1];
    startindex = atoi(argv[2]);

    /** MPI who am I and who is root here **/
    const MPI_Comm comm = MPI_COMM_WORLD;
    int mpi_size;
    int mpi_rank;
    const int mpi_root = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    clock_t start_t, end_t;
    double total_t;
    if (mpi_rank == mpi_root){
        start_t = clock();
        printf("Starting of the program, start_t = %ld\n", start_t);
    }

    int sz, ntodo;
    int N_SN;

    int  OFFDIAG, N_START, N_END, N_USED;
    double w0, wa, omega_m, A, h, A_k005;
    double  k, n, z;
    // FILE *ifp;

    /* Containers for vectors used by MPI */
    /* Everything */
    int *i_all, *j_all;
    double *SN_z_i_all, *SN_th_i_all,*SN_ph_i_all;
    double *SN_z_j_all, *SN_th_j_all,*SN_ph_j_all;
    double *ans_all;
    unsigned int *todo_all;

    /* Everything left to do */
    int *i_left, *j_left;
    double *SN_z_i_left, *SN_th_i_left,*SN_ph_i_left;
    double *SN_z_j_left, *SN_th_j_left,*SN_ph_j_left;
    double *ans_left;
    unsigned int *todo_left;

    /* ones assigned to a rank */
    int *i_loc, *j_loc;
    double *SN_z_i_loc, *SN_th_i_loc, *SN_ph_i_loc, *SN_z_j_loc, *SN_th_j_loc, *SN_ph_j_loc, *ans_loc;
    unsigned int *todo_loc;
    int* randomorder;   /* used to scramble integrals for load balancing */

    gsl_set_error_handler_off();

    PRINT_FLAG = 0;
    /** Parameters used in cosmological simulation **/
    omega_m=0.286; 

    n=0.965;  

    omhh=0.140;   
    obhh=0.0223; 

    h=sqrt(omhh/omega_m);
    A_k005 = 2.0e-9; 

    A    = A_k005;

    // only for my own P(k) funcs in functions.c
    A_k0002 = A_k005 / pow(25.0, n-1);

    H0_mpcinv  = h/2997.9;   /* in Mpc-1; need this for ps_linear */
    /**initialize power spectrum **/
    w0=-1.0;
    wa=0.0;  
    
    /***Spline the r(z) ***/    
    spline_r(omega_m, w0, wa);

    /**Spline the T(k) evaluated using CAMB  do_nonlin=1 HERE, and dn/dlnk=0  **/
    if (mpi_rank == mpi_root){
        run_camb_get_Tk_friendly_format(1, omega_m, omhh, obhh, n, 0.0, A, w0); 
    }
    MPI_Barrier(comm);
    WINDOW = 1;    /* Mean subtracted using int over all-sky (DEFAULT) */
    spline_Tk_from_CAMB(omega_m, w0, wa, A, n);    
    spline_Pk_from_CAMB   (0, omega_m, w0, wa, A, n);    
    spline_Pk_from_CAMB_NR(0, omega_m, w0, wa, A, n);    
    
    /** Spline the P(k) evaluated at z=0 then later rescale it by the growth factor **/

    NSPLINE=0;
    for(k=K_START_INTERP; k<=K_END_INTERP; k+=DLNK_PKSPLINE*k) NSPLINE++;    
        z=0;
    spline_Pk(z, omega_m, w0, wa, A_k0002, n);
    if (mpi_rank == mpi_root){ 
        printf("D(z=0)=%f, D(z=1)=%f, D(z=2)=%f\n", D_tab(0.), D_tab(1.0), D_tab(2.0));  
        printf("the raw sigma8 = %f\n", sigma8(0));
        printf("growth_k0_new  = %f\n", growth_k0_new);
    }

    /*********************************************************************/
    /*** Tabulate the jl'(x) function using both GSL and NR             **/
    /** latter gives the same results, and is required for parallel run **/
    /*********************************************************************/
    LMAX_CIJ_THETA=200; // needs 200-300 for higher-z (z>0.05 or so) convergence even for diag elements - Oct 2016

    spline_jlprime_range_ell   (LMAX_CIJ_THETA); // can't use for parallel
    spline_jlprime_range_ell_NR(LMAX_CIJ_THETA); // need for parallel runs


    int* todoindeces;
  // Allocate master array only on master MPI rank
    if (mpi_rank == mpi_root){


        /*******************************************************************/
        /**   get #SN and define the SN data and covariance mat arrays   ***/
        /*******************************************************************/

        double **all_SN_z_th_phi, **all_Noise_SN, *all_delta_m;
        double **SN_z_th_phi, **Signal_SN,  **Noise_SN, *delta_m;
        char SN_filename[256];

        sprintf(SN_filename , "%s.dat",fileroot);

            // sprintf(SN_filename , "small_data_clean.txt");

            /************************/
            /*** fiducial choices ***/
            /************************/
        
        OFFDIAG=1;     /*full off-diag Noise */
        N_START = 1;   /* hardcoded before already */
        N_END=file_eof_linecount(SN_filename);       /* How many SN would you actually like to USE? DEFAULT=208, but choice doesntmatter */
        N_USED = N_END-N_START+1;    


        /***********************************************************/
        /***   select datafiles and define some arrays/matrices  ***/
        /***********************************************************/
        N_SN = file_eof_linecount(SN_filename);
        printf("I see %d SN in that file\n", N_SN);

        all_SN_z_th_phi = dmatrix(1, N_SN, 1, 4);
        all_Noise_SN  = dmatrix(1, N_SN, 1, N_SN);
        all_delta_m   = dvector(1, N_SN);
        
        // SN_z_th_phi = dmatrix(1, N_USED, 1, 4);
        // Noise_SN  = dmatrix(1, N_USED, 1, N_USED);
        // delta_m   = dvector(1, N_USED);

        /***********************************************************/
        /** read the SN from file, possibly with noise covariance **/
        /***********************************************************/
        read_pos_noise(OFFDIAG, all_SN_z_th_phi, all_delta_m, all_Noise_SN, SN_filename);

        /**********************************************/
        /** Evaluate the signal matrix of PanStarrs ***/
        /**********************************************/
        Signal_SN = dmatrix(1, N_SN, 1, N_SN);        
        // COUNT_IN=0; COUNT_BELOW=0;  COUNT_ABOVE=0;

        /****/
        /** Vectorize output into something useful for MPI **/
        /****/


        sz = N_SN * (N_SN+1)/2;
        i_all = malloc(sz*sizeof(int));
        j_all = malloc(sz*sizeof(int));
        SN_z_i_all= malloc(sz*sizeof(double));
        SN_th_i_all= malloc(sz*sizeof(double));
        SN_ph_i_all= malloc(sz*sizeof(double));
        SN_z_j_all= malloc(sz*sizeof(double));
        SN_th_j_all= malloc(sz*sizeof(double));
        SN_ph_j_all= malloc(sz*sizeof(double));
        ans_all = malloc(sz*sizeof(double));
        todo_all = malloc(sz*sizeof(unsigned int));

        /* Randomize the order of elements in the array for load balancing */
        struct Orderer* orders =  malloc(sz*(sizeof *orders) );

        srand(1);
        int index=0;
        for(int ii=0; ii< N_SN; ii++){
            for(int jj=ii; jj< N_SN; jj++){
                orders[index].index = index;
                orders[index].comparator = rand();
                // orders[index].comparator = index;
                index++;
            }  
        }
        qsort(orders, sz, sizeof(*orders), cmpfunc);
        randomorder = malloc(sz*sizeof(int));
        for (int ii=0; ii<sz;ii++){
            randomorder[ii]=orders[ii].index;
        }

        index=0;
        for(int ii=0; ii< N_SN; ii++){
            for(int jj=ii; jj< N_SN; jj++){
                i_all[randomorder[index]] = ii;
                j_all[randomorder[index]] = jj;
                SN_z_i_all[randomorder[index]]= all_SN_z_th_phi[ii+1][1];
                SN_th_i_all[randomorder[index]]= all_SN_z_th_phi[ii+1][2];
                SN_ph_i_all[randomorder[index]]= all_SN_z_th_phi[ii+1][3];
                SN_z_j_all[randomorder[index]]= all_SN_z_th_phi[jj+1][1];
                SN_th_j_all[randomorder[index]]= all_SN_z_th_phi[jj+1][2];
                SN_ph_j_all[randomorder[index]]= all_SN_z_th_phi[jj+1][3];
                todo_all[index]=1;
                index++;
            }  
        }

        if (startindex != 0){
            char SN_filename[256];
            FILE * ptr;
            sprintf(SN_filename , "%s.xi.%d",fileroot,startindex);
            ptr = fopen(SN_filename,"rb");  // r for read, b for binary
            fread(ans_all,sizeof(double),sz,ptr); // read 10 bytes to our buffer
            fclose(ptr);
            sprintf(SN_filename , "%s.xitodo.%d",fileroot,startindex);
            ptr = fopen(SN_filename,"rb");  // r for read, b for binary
            fread(todo_all,sizeof(unsigned int),sz,ptr); // read 10 bytes to our buffer
            fclose(ptr);

            ntodo=0;
            for (int jj = 0; jj<sz ; jj++) ntodo+=todo_all[jj];   

            /* now prune the big list */
            i_left = malloc(ntodo*sizeof(int));
            j_left = malloc(ntodo*sizeof(int));
            SN_z_i_left= malloc(ntodo*sizeof(double));
            SN_th_i_left= malloc(ntodo*sizeof(double));
            SN_ph_i_left= malloc(ntodo*sizeof(double));
            SN_z_j_left= malloc(ntodo*sizeof(double));
            SN_th_j_left= malloc(ntodo*sizeof(double));
            SN_ph_j_left= malloc(ntodo*sizeof(double));
            ans_left = malloc(ntodo*sizeof(double));
            todo_left = malloc(ntodo*sizeof(unsigned int));
            todoindeces = malloc(ntodo*sizeof(int));
            index=0;
            for (int jj=0; jj<sz ; jj++){
                if (todo_all[jj]) {
                    i_left[index] = i_all[jj];
                    j_left[index] = j_all[jj];
                    SN_z_i_left[index]=SN_z_i_all[jj];
                    SN_th_i_left[index]=SN_th_i_all[jj];
                    SN_ph_i_left[index]=SN_ph_i_all[jj];
                    SN_z_j_left[index]=SN_z_j_all[jj];
                    SN_th_j_left[index]= SN_th_j_all[jj];
                    SN_ph_j_left[index]= SN_ph_j_all[jj];
                    ans_left[index] = ans_all[jj];
                    todo_left[index] = todo_all[jj];
                    todoindeces[index] = jj;
                    index++;
                }
            }
            free(i_all);
            free(j_all);
            free(SN_z_i_all);
            free(SN_th_i_all);
            free(SN_ph_i_all);
            free(SN_z_j_all);
            free(SN_th_j_all);
            free(SN_ph_j_all);
        } else{
            ntodo=sz;
            todoindeces = malloc(ntodo*sizeof(int));
            for (int jj=0;jj<ntodo;jj++) {
                todoindeces[jj]=jj;
            }
            i_left = i_all;
            j_left = j_all;
            SN_z_i_left= SN_z_i_all;
            SN_th_i_left= SN_th_i_all;
            SN_ph_i_left= SN_ph_i_all;
            SN_z_j_left= SN_z_j_all;
            SN_th_j_left= SN_th_j_all;
            SN_ph_j_left= SN_ph_j_all;
            ans_left = ans_all;
            todo_left = todo_all;
        }
        printf("Number to do %d\n",ntodo); 
    }
    
    MPI_Bcast(&N_SN, 1, MPI_INT, mpi_root, comm);
    MPI_Bcast(&sz, 1, MPI_INT, mpi_root, comm);
    MPI_Bcast(&ntodo, 1, MPI_INT, mpi_root, comm);


    /****/
    /** for full vectorized list of inputs how much do I send and from where **/
    /****/

    int *sendcounts, *displs, recvcount;
    int remainder;

    // Because the number of MPI ranks may not divide evenly into the size of the
    // master array, we must allocate an array which contains the number of
    // master array elements to send to each MPI rank, which may differ across
    // MPI ranks.
    sendcounts = malloc(mpi_size*sizeof(int));
    // This is the displacement from the zeroth element of the master array which
    // corresponds to the ith chunk of the master array which is scattered to the
    // ith MPI rank.
    displs = malloc(mpi_size*sizeof(int));

    // Set up array chunks to send to each MPI process. If the number of MPI
    // processes does not divide evenly into the number of elements in the array,
    // distribute the remaining parts among the first several MPI ranks in order
    // to achieve the best possible load balance.

    remainder = ntodo % mpi_size;
    for (int i = 0; i < mpi_size; i++) sendcounts[i] = ntodo / mpi_size;
    for (int i = 0; i < remainder; i++) sendcounts[i]++;
    displs[mpi_root] = 0;
    for (int i = 1; i < mpi_size; i++) displs[i] = displs[i-1] + sendcounts[i-1];

    i_loc = malloc(sendcounts[mpi_rank]*sizeof(int));
    j_loc = malloc(sendcounts[mpi_rank]*sizeof(int));
    SN_z_i_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    SN_th_i_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    SN_ph_i_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    SN_z_j_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    SN_th_j_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    SN_ph_j_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    ans_loc = malloc(sendcounts[mpi_rank]*sizeof(double));
    todo_loc = malloc(sendcounts[mpi_rank]*sizeof(unsigned int));

    recvcount = sendcounts[mpi_rank];

// Scatter the master array on the master MPI rank to various MPI ranks.
    MPI_Scatterv(i_left, sendcounts, displs,
       MPI_INT, i_loc, recvcount,
       MPI_INT, 0, comm);
    MPI_Scatterv(j_left, sendcounts, displs,
       MPI_INT, j_loc, recvcount,
       MPI_INT, 0, comm);
    MPI_Scatterv(SN_z_i_left, sendcounts, displs,
       MPI_DOUBLE, SN_z_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_th_i_left, sendcounts, displs,
       MPI_DOUBLE, SN_th_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_ph_i_left, sendcounts, displs,
       MPI_DOUBLE, SN_ph_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_z_j_left, sendcounts, displs,
       MPI_DOUBLE, SN_z_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_th_j_left, sendcounts, displs,
       MPI_DOUBLE, SN_th_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_ph_j_left, sendcounts, displs,
       MPI_DOUBLE, SN_ph_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(ans_left, sendcounts, displs,
       MPI_DOUBLE, ans_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(todo_left, sendcounts, displs,
       MPI_UNSIGNED, todo_loc, recvcount,
       MPI_UNSIGNED, 0, comm);

    int maxsendcounts = -1;
    for (int i = 0; i < mpi_size; i++) {
        if (sendcounts[i] > maxsendcounts){
            maxsendcounts = sendcounts[i];
        }
    }

    int savefreq=250;
    int base = 0;
    int nloops = maxsendcounts/savefreq;
    if (maxsendcounts % savefreq  !=0) nloops=nloops+1;
    // printf("%d %d %d \n",maxsendcounts,savefreq,nloops);

    for (int ii=0; ii<  nloops; ii++){

        int ncts=0;
        // for (int jj=0;jj<recvcount;jj++) printf("%d ",todo_loc[jj]);
        // printf("\n");
        if ((recvcount - base) >= savefreq){
            ncts = savefreq;
        } else {
            ncts = recvcount-base;
        }
        if ((recvcount - base) >=0){
            calculate_Cov_vel_of_SN_vec(ncts, &i_loc[base], &j_loc[base],
                &SN_z_i_loc[base], &SN_th_i_loc[base],&SN_ph_i_loc[base], &SN_z_j_loc[base], &SN_th_j_loc[base],&SN_ph_j_loc[base], 
                &ans_loc[base], omega_m, w0, wa);
            // for (int jj =0; jj<ncts;jj++) ans_loc[base+jj]=100*i_loc[base+jj]+ j_loc[base+jj];
            for (int jj =0;jj<ncts;jj++) todo_loc[base+jj]=0;
        }

        // calculate_Cov_vel_of_SN_vec(recvcount, i_loc, j_loc,
        //     SN_z_i_loc, SN_th_i_loc,SN_ph_i_loc, SN_z_j_loc, SN_th_j_loc,SN_ph_j_loc, 
        //     ans_loc, omega_m, w0, wa);

        base = base+ ncts;
        MPI_Gatherv(ans_loc, recvcount, MPI_DOUBLE, ans_left, sendcounts, displs,
            MPI_DOUBLE, 0, comm);
        MPI_Gatherv(todo_loc, recvcount, MPI_UNSIGNED, todo_left, sendcounts, displs,
            MPI_UNSIGNED, 0, comm);

        if (mpi_rank == mpi_root){
            for (int jj = 0; jj<ntodo;jj++) {
                ans_all[todoindeces[jj]] = ans_left[jj];
            }
            int dum=0;
            for (int jj = 0; jj<ntodo ; jj++) dum+=todo_left[jj]; 
            char SN_filename[256];
            sprintf(SN_filename , "%s.xi.%d",fileroot,dum);
            FILE *f = fopen(SN_filename, "wb");
            fwrite(ans_all, sizeof(double), sz, f);
            fclose(f);
            sprintf(SN_filename , "%s.xitodo.%d",fileroot,dum);
            f = fopen(SN_filename, "wb");
            fwrite(todo_all, sizeof(unsigned int), sz, f);
            fclose(f);
        }
    }
 
    MPI_Gatherv(ans_loc, recvcount, MPI_DOUBLE, ans_left, sendcounts, displs,
       MPI_DOUBLE, 0, comm);

    if (mpi_rank == mpi_root){            

        for (int jj = 0; jj<ntodo;jj++) {
            ans_all[todoindeces[jj]] = ans_left[jj];
        }

        /* redistribute answers */
        double *ans_resort;
        ans_resort = malloc(sz*sizeof(double));
        for (int ii=0; ii<sz;ii++){
            ans_resort[ii]= ans_all[randomorder[ii]];
        }
        char SN_filename[256];
        sprintf(SN_filename , "%s.xi",fileroot);
        FILE *f = fopen(SN_filename, "wb");
        fwrite(ans_resort, sizeof(double), sz, f);
        fclose(f);
        end_t = clock();
        printf("End of the big loop, end_t = %ld\n", end_t);
        total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        printf("Total time taken by CPU: %f\n", total_t  );
    }
    MPI_Finalize();
    exit(0);
}
