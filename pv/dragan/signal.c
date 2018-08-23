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


int main()
{
    /** MPI who am I and who is root here **/
    const MPI_Comm comm = MPI_COMM_WORLD;
    int mpi_size;
    int mpi_rank;
    const int mpi_root = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    int sz;
    int N_SN;

    int  OFFDIAG, N_START, N_END, N_USED;
    double w0, wa, omega_m, A, h, A_k005;
    double  k, n, z;
    FILE *ifp;

    /* Containers for vectors used by MPI */
    int *i_all, *j_all;
    double *SN_z_i_all, *SN_th_i_all,*SN_ph_i_all;
    double *SN_z_j_all, *SN_th_j_all,*SN_ph_j_all;
    double *ans_all;
    int *i_loc, *j_loc;
    double *SN_z_i_loc, *SN_th_i_loc, *SN_ph_i_loc, *SN_z_j_loc, *SN_th_j_loc, *SN_ph_j_loc, *ans_loc;

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



  // Allocate master array only on master MPI rank
    if (mpi_rank == mpi_root){

        /*******************************************************************/
        /**   get #SN and define the SN data and covariance mat arrays   ***/
        /*******************************************************************/

        double **all_SN_z_th_phi, **all_Noise_SN, *all_delta_m;
        double **SN_z_th_phi, **Signal_SN,  **Noise_SN, *delta_m;
        char SN_filename[256];
        sprintf(SN_filename , "pvlist.1234.dat");
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

        int index=0;
        for(int ii=0; ii< N_SN; ii++){
            for(int jj=ii; jj< N_SN; jj++){
                i_all[index] = ii;
                j_all[index] = jj;
                SN_z_i_all[index]= all_SN_z_th_phi[ii+1][1];
                SN_th_i_all[index]= all_SN_z_th_phi[ii+1][2];
                SN_ph_i_all[index]= all_SN_z_th_phi[ii+1][3];
                SN_z_j_all[index]= all_SN_z_th_phi[jj+1][1];
                SN_th_j_all[index]= all_SN_z_th_phi[jj+1][2];
                SN_ph_j_all[index]= all_SN_z_th_phi[jj+1][3];
                index++;
            }  
        }
    }
    
    MPI_Bcast(&N_SN, 1, MPI_INT, mpi_root, comm);
    MPI_Bcast(&sz, 1, MPI_INT, mpi_root, comm);

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

    remainder = sz % mpi_size;
    for (int i = 0; i < mpi_size; i++) sendcounts[i] = sz / mpi_size;
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

    recvcount = sendcounts[mpi_rank];

// Scatter the master array on the master MPI rank to various MPI ranks.
    MPI_Scatterv(i_all, sendcounts, displs,
       MPI_INT, i_loc, recvcount,
       MPI_INT, 0, comm);
    MPI_Scatterv(j_all, sendcounts, displs,
       MPI_INT, j_loc, recvcount,
       MPI_INT, 0, comm);
    MPI_Scatterv(SN_z_i_all, sendcounts, displs,
       MPI_DOUBLE, SN_z_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_th_i_all, sendcounts, displs,
       MPI_DOUBLE, SN_th_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_ph_i_all, sendcounts, displs,
       MPI_DOUBLE, SN_ph_i_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_z_j_all, sendcounts, displs,
       MPI_DOUBLE, SN_z_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_th_j_all, sendcounts, displs,
       MPI_DOUBLE, SN_th_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);
    MPI_Scatterv(SN_ph_j_all, sendcounts, displs,
       MPI_DOUBLE, SN_ph_j_loc, recvcount,
       MPI_DOUBLE, 0, comm);

    calculate_Cov_vel_of_SN_vec(recvcount, i_loc, j_loc,
        SN_z_i_loc, SN_th_i_loc,SN_ph_i_loc, SN_z_j_loc, SN_th_j_loc,SN_ph_j_loc, 
        ans_loc, omega_m, w0, wa);
    // printf("%2d %2d %2d %2d \n",mpi_rank,recvcount,sendcounts[mpi_rank], displs[mpi_rank]);
    MPI_Gatherv(ans_loc, recvcount, MPI_DOUBLE, ans_all, sendcounts, displs,
       MPI_DOUBLE, 0, comm);

    if (mpi_rank == mpi_root){ 
// calculate_Cov_vel_of_SN(N_SN, all_SN_z_th_phi, Signal_SN, omega_m, w0, wa);

//        printf("%e %e\n", ans_all[0], ans_all[1]);
// printf("%e %e\n", Noise_SN[1][1], Noise_SN[N_USED][N_USED]);

        FILE *f = fopen("client.data", "wb");
        fwrite(ans_all, sizeof(double), sz, f);
        fclose(f);

// ifp=fopen("pvlist.1234.xi", "w");
// for(i=1;i<=N_SN;i++){
//     for(j=1;j<=N_SN;j++)
//         fprintf(ifp,"%0.9e ",Signal_SN[i][j]);
//     fprintf(ifp,"\n");
// }
// fclose(ifp);
    }
    exit(0);
}
