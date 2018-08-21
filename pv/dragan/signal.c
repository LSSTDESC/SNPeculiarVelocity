#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include<omp.h>
#include"declarations.h"
#include"recipes/nrutil.h"
#include"recipes/nr.h"


int main()
{
    int  OFFDIAG, N_START, N_END, N_USED;
    int i, j, jj;
    double w0, wa, omega_m, A, h, A_k005;
    double  k, n, z;
    FILE *ifp;

    gsl_set_error_handler_off();

    PRINT_FLAG = 0;
    /**************************************/
    /** Planck parameters - August 2016! **/
    /**************************************/
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
    /********************************/
    /** initialize power spectrum ***/
    /********************************/
    w0=-1.0;
    wa=0.0;  
    
    /*********************/
    /** Spline the r(z) **/
    /*********************/    
    spline_r(omega_m, w0, wa);

    /************************************************/
    /** Spline the T(k) evaluated using CAMB       **/
    /** do_nonlin=1 HERE, and dn/dlnk=0            **/
    /************************************************/
    run_camb_get_Tk_friendly_format(1, omega_m, omhh, obhh, n, 0.0, A, w0);
    spline_Tk_from_CAMB(omega_m, w0, wa, A, n);    
    spline_Pk_from_CAMB   (0, omega_m, w0, wa, A, n);    
    spline_Pk_from_CAMB_NR(0, omega_m, w0, wa, A, n);    
    
    /************************************************/
    /** Spline the P(k) evaluated at z=0           **/
    /** then later rescale it by the growth factor **/
    /************************************************/

    NSPLINE=0;
    for(k=K_START_INTERP; k<=K_END_INTERP; k+=DLNK_PKSPLINE*k) NSPLINE++;    
    z=0;
    spline_Pk(z, omega_m, w0, wa, A_k0002, n);

    printf("D(z=0)=%f, D(z=1)=%f, D(z=2)=%f\n", D_tab(0.), D_tab(1.0), D_tab(2.0));  
    printf("the raw sigma8 = %f\n", sigma8(0));
    printf("growth_k0_new  = %f\n", growth_k0_new);

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
    WINDOW = 1;    /* Mean subtracted using int over all-sky (DEFAULT) */

    /*********************************************************************/
    /*** Tabulate the jl'(x) function using both GSL and NR             **/
    /** latter gives the same results, and is required for parallel run **/
    /*********************************************************************/
    LMAX_CIJ_THETA=200; // needs 200-300 for higher-z (z>0.05 or so) convergence even for diag elements - Oct 2016

    spline_jlprime_range_ell   (LMAX_CIJ_THETA); // can't use for parallel
    spline_jlprime_range_ell_NR(LMAX_CIJ_THETA); // need for parallel runs

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
    COUNT_IN=0; COUNT_BELOW=0;  COUNT_ABOVE=0;

    calculate_Cov_vel_of_SN(N_SN, all_SN_z_th_phi, Signal_SN, omega_m, w0, wa);

    printf("%e %e\n", Signal_SN[1][1], Signal_SN[N_USED][N_USED]);
    // printf("%e %e\n", Noise_SN[1][1], Noise_SN[N_USED][N_USED]);

    FILE *f = fopen("client.data", "wb");
    for(i=1;i<=N_SN;i++){
            fwrite(Signal_SN[i]+1, sizeof(double), N_SN, f);
    }
    fclose(f);

    ifp=fopen("pvlist.1234.xi", "w");
    for(i=1;i<=N_SN;i++){
        for(j=1;j<=N_SN;j++)
            fprintf(ifp,"%0.9e ",Signal_SN[i][j]);
        fprintf(ifp,"\n");
    }
    fclose(ifp);
    exit(0);
}
