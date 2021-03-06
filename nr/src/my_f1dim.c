#include "nrutil.h"

extern int ncom;
extern double *pcom,*xicom,(*nrfunc)(double []);

double my_f1dim(double x)
{
	int j;
	double f,*xt;

	xt=dvector(1,ncom);
	for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
	f=(*nrfunc)(xt);
	free_dvector(xt,1,ncom);
	return f;
}
