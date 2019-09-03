#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker, cm
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2.0

# cosmology
OmegaM0 = 0.28
cosmo = FlatLambdaCDM(H0=100, Om0=OmegaM0)

gamma=0.55

sigOM0sqinv=1/0.005**2

# power spectrum from CAMB
matter = numpy.loadtxt("/Users/akim/project/PeculiarVelocity/pv/dragan/matterpower.dat")

# vv
f=OmegaM0**0.55

# SN properties
restrate_Ia = 0.65*  2.69e-5*(1/0.70)**3 # h^3/Mpc^3 yr^{-1}
sigm_Ia = 0.08

# galaxy density (TAIPAN Howlett et al.)
ng = 1e-3

def Cinverse(m00,m11,m01):
    return numpy.array([[m11,-m01],[-m01,m00]])/(m00*m11 - m01**2)

def Cinverse_partial(m00,m11,m01,dm00,dm11,dm01):
    den = (m00*m11 - m01**2)
    return numpy.array([[dm11,2*m01*dm01],[2*m01*dm01,dm00]])/den \
        - numpy.array([[m11,-m01],[-m01,m00]])/den**2 * (dm00*m11 + m00*dm11 - 2*m01*dm01)

def OmegaM(a,OmegaM0=OmegaM0):
    return OmegaM0/(OmegaM0 + (1-OmegaM0)*a**3)

def integrand_dlnfs(a,*args):
    return numpy.log(OmegaM(a,*args))* OmegaM(a,*args)**0.55 / a

def dlnfs8dg(a,*args):
    return numpy.log(OmegaM(a,*args)) + integrate.quad(integrand_dlnfs, 1e-8, a)[0]

def integrand_D(a,*args):
    return  OmegaM(a,*args)**0.55 / a

# normalized to a=1
def D(a,*args):
    return numpy.exp(-integrate.quad(integrand_D, a,1,args=args)[0])

def dOmdOm0(a,OmegaM0=OmegaM0):
    den= (OmegaM0 + (1-OmegaM0)*a**3) 
    return (1./den - (1-a**3)/den**2)

def integrand_dDdOmOverD(a,*args):
    return  gamma*OmegaM(a,*args)**(gamma-1) / a

def dDdOmOverD(a, *args):
    return integrate.quad(integrand_dDdOmOverD, a,1,args=args)[0]


def dfdOm(a):
    return gamma * OmegaM(a)**(gamma-1)



def Pvv(mu,f,D):
    return (f*D*mu*100)**2*matter[:,1]/matter[:,0]**2

def Pvv_l(mu,f,D):
    return 2*f*D*(mu*100)**2*matter[:,1]/matter[:,0]**2

def Pvv_Om(mu,f,D, dfdOm, dDdOmOverD):
    return 2*f*D*(mu*100)**2*matter[:,1]/matter[:,0]**2 * (f*dDdOmOverD*D + dfdOm * D )


# gg
b= 1.2
def Pgg(mu,f,D):
    return (b*D+f*D*mu**2)**2*matter[:,1]

def Pgg_l(mu,f,D):
    return 2*mu**2*(b*D+f*D*mu**2)*matter[:,1]

def Pgg_b(mu,f,D):
    return 2*(b*D+f*D*mu**2)*matter[:,1]

def Pgg_Om(mu,f,D, dfdOm, dDdOmOverD):
    return 2*(b*D+f*D*mu**2)*matter[:,1] * (b*dDdOmOverD*D + f* dDdOmOverD*D*mu**2 + dfdOm * D*mu**2 )


# gv
def Pgv(mu,f,D):
    return (b*D+f*D*mu**2)*(f*D*mu*100)*matter[:,1]/matter[:,0]

def Pgv_l(mu,f,D):
    return (mu**2*(f*D*mu*100)+(b*D+f*D*mu**2)*(mu*100))*matter[:,1]/matter[:,0]

def Pgv_b(mu,f,D):
    return (f*D*mu*100)*matter[:,1]/matter[:,0]

def Pgv_Om(mu,f,D, dfdOm, dDdOmOverD):
    return ((b*dDdOmOverD*D + f*dDdOmOverD*D*mu**2 + dfdOm*D*mu**2) * (f*D*mu*100) +(b*D+f*D*mu**2) *mu*100*(f*dDdOmOverD*D + dfdOm*D) )*matter[:,1]/matter[:,0]


def finvp(f00,f11,f01,f00p,f11p,f01p):
    den = f00*f11 -f01**2
    return f11p /den - f11*(f00p*f11 + f00*f11p - 2*f01*f01p)/den**2

def finvp3d(f00,f01,f02,f11,f12,f22,f00p,f01p,f02p,f11p,f12p,f22p):
    den = f00*f11*f22 - f00*f12**2-f11*f02**2-f22*f01**2 + 2*f01*f02*f12
    denp = f00p*f11*f22 + f00*f11p*f22+f00*f11*f22p  \
        - f00p*f12**2 - 2*f00*f12*f12p \
        - f11p*f02**2 - 2*f11*f02*f02p \
        - f22p*f01**2 - 2* f22*f01*f01p \
        + 2*(f01p*f02*f12 + f01*f02p*f12 + f01*f02*f12p)

    # full matrix
    # mat = numpy.array([ \
    #     [f11*f22-f12**2, f02*f12-f01*f22, f01*f12-f00*f12], \
    #     [0, f00*f22-f02**2, f02*f01-f02*f11], \
    #     [0, 0, f00*f11-f01**2] \
    #     ])
    #00 element
    mat = f11*f22-f12**2

    # mat[1,0]= mat[0,1]
    # mat[2,0] = mat[0,2]
    # mat[2,1] = mat[1,2]

    #full matrix
    #00 element
    # matp = numpy.array([ \
    #     [f11*f22p+f11p*f22-2*f12*f12p, f02p*f12+f02*f12p-f01p*f22-f01*f22p, f01p*f12+f01*f12p-f00p*f12-f00*f12p], \
    #     [0, f00p*f22+f00*f22p-2*f02*f02p, f02p*f01+f02*f01p-f02p*f11-f02*f11p], \
    #     [0, 0, f00p*f11+f00*f11p-2*f01*f01p] \
    #     ])
    matp =f11*f22p+f11p*f22-2*f12*f12p

    # matp[1,0]= matp[0,1]
    # matp[2,0] = matp[0,2]
    # matp[2,1] = matp[1,2]

    return matp/den -denp/den**2*mat


def Cmatrices(z,mu,ng,duration,sigm,restrate):
    a = 1/(1.+z)
    OmegaM_a = OmegaM(a)
    n = duration*restrate/(1+z)
    sigv_factor = numpy.log(10)/5*z/(1+z)
    sigv =sigm * sigv_factor *3e5
    lnfgfactor = dlnfs8dg(a)

    sigv2 = sigv**2
    nvinv = sigv2/n
    ninv = 1./n

    # nginv = 1./ng

    f = OmegaM(a)**.55
    D_ = D(a)
    dfdOm_ = dfdOm(a)
    dDdOmOverD_ = dDdOmOverD(a)

    pggs = Pgg(mu,f,D_)
    pgvs = Pgv(mu,f,D_)
    pvvs = Pvv(mu,f,D_)

    pggs_l = Pgg_l(mu,f,D_)
    pgvs_l = Pgv_l(mu,f,D_)
    pvvs_l = Pvv_l(mu,f,D_)

    pggs_b = Pgg_b(mu,f,D_)
    pgvs_b = Pgv_b(mu,f,D_)

    dOmdOm0_ = dOmdOm0(a,OmegaM0=OmegaM0)
    pggs_Om = Pgg_Om(mu,f,D_,dfdOm_,dDdOmOverD_)*dOmdOm0_
    pgvs_Om = Pgv_Om(mu,f,D_,dfdOm_,dDdOmOverD_)*dOmdOm0_
    pvvs_Om = Pvv_Om(mu,f,D_,dfdOm_,dDdOmOverD_)*dOmdOm0_

    C=[]
    Cinv = []
    dCdl = []
    dCdb = []
    dCdOm = []
    # Cinvs = []
    Cinvn = []
    Cinvsigmam = []
    for pgg, pgv, pvv, pgg_l, pgv_l, pvv_l, pgg_b, pgv_b, pgg_Om, pgv_Om, pvv_Om  in zip(pggs,pgvs,pvvs,pggs_l,pgvs_l,pvvs_l,pggs_b,pgvs_b,pggs_Om,pgvs_Om,pvvs_Om):
        C.append(numpy.array([[pgg+ninv,pgv],[pgv,pvv+nvinv]]))
        Cinv.append(numpy.linalg.inv(C[-1]))
        dCdl.append(numpy.array([[pgg_l,pgv_l],[pgv_l,pvv_l]])*OmegaM_a**0.55*lnfgfactor) # second term converts from fsigma8 to gamma
        dCdb.append(numpy.array([[pgg_b,pgv_b],[pgv_b,0]]))
        dCdOm.append(numpy.array([[pgg_Om,pgv_Om],[pgv_Om,pvv_Om]]))

        den = (pgg+ninv)*(pvv+nvinv)-pgv**2
        # Cinvs.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+nginv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+nginv]]))
        # Cinvs[-1]=-Cinvs[-1]*sigv**2/restrate/duration**2 #in terms of lnt not s = sigma^2/rate/s

        # Cinvn.append(-1./den * numpy.array([[sigv2,0],[0,1]]) + (sigv2*(pgg+ninv)+(pvv+nvinv))/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))
        # Cinvn.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 1, sigv2,0)) # partial with respect to 1/n

        # Cinvn[-1]=Cinvn[-1]/n #in terms of lnt  n**-2 * n = 1/n

        Cinvn.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, -ninv, -sigv2*ninv,0)) # partial wrt lnt
        # Cinvsigmam.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+ninv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))

        # print (Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 0, 2*sigv*ninv*sigv_factor *3e5 ,0))
        # Cinvsigmam[-1] = Cinvsigmam[-1] * sigv_factor *3e5 * 2 * sigv / n
        Cinvsigmam.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 0, 2*sigv*ninv*sigv_factor *3e5 ,0)) # partial wrt sigmaM

       
    return C, Cinv, dCdl,dCdb,Cinvn, Cinvsigmam,dCdOm

def traces(z,mu,ng,duration,sigm,restrate):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm,restrate)

    ll=[]
    lb=[]
    bb=[]
    lO=[]
    bO=[]
    OO=[]

    lls=[]
    lbs=[]
    bbs=[]
    lOs=[]
    bOs=[]
    OOs=[]


    ll_ind=[]
    lb_ind=[]
    bb_ind=[]
    lO_ind=[]
    bO_ind=[]
    OO_ind=[]

    llsigM=[]
    lbsigM=[]
    bbsigM=[]
    lOsigM=[]
    bOsigM=[]
    OOsigM=[]


    ll_vonly=[]
    lO_vonly=[]
    OO_vonly=[]

    for C, Cinv, C_l,C_b,Cinvn,CinvsigM, C_Om in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4],cmatrices[5],cmatrices[6]):
        ll.append(numpy.trace(Cinv@C_l@Cinv@C_l))
        bb.append(numpy.trace(Cinv@C_b@Cinv@C_b))
        lb.append(numpy.trace(Cinv@C_l@Cinv@C_b))
        lO.append(numpy.trace(Cinv@C_l@Cinv@C_Om))
        bO.append(numpy.trace(Cinv@C_b@Cinv@C_Om))
        OO.append(numpy.trace(Cinv@C_Om@Cinv@C_Om))

        lls.append(numpy.trace(Cinvn@ C_l@Cinv@C_l+Cinv@ C_l@Cinvn@C_l))
        bbs.append(numpy.trace(Cinvn@ C_b@Cinv@C_b+Cinv@ C_b@Cinvn@C_b))
        lbs.append(numpy.trace(Cinvn@ C_l@Cinv@C_b+Cinv@ C_l@Cinvn@C_b))
        lOs.append(numpy.trace(Cinvn@ C_l@Cinv@C_Om+Cinv@ C_l@Cinvn@C_Om))
        bOs.append(numpy.trace(Cinvn@ C_b@Cinv@C_Om+Cinv@ C_b@Cinvn@C_Om))
        OOs.append(numpy.trace(Cinvn@ C_Om@Cinv@C_Om+Cinv@ C_Om@Cinvn@C_Om))

        llsigM.append(numpy.trace(CinvsigM@ C_l@Cinv@C_l+Cinv@ C_l@CinvsigM@C_l))
        bbsigM.append(numpy.trace(CinvsigM@ C_b@Cinv@C_b+Cinv@ C_b@CinvsigM@C_b))
        lbsigM.append(numpy.trace(CinvsigM@ C_l@Cinv@C_b+Cinv@ C_l@CinvsigM@C_b))
        lOsigM.append(numpy.trace(CinvsigM@ C_l@Cinv@C_Om+Cinv@ C_l@CinvsigM@C_Om))
        bOsigM.append(numpy.trace(CinvsigM@ C_b@Cinv@C_Om+Cinv@ C_b@CinvsigM@C_Om))
        OOsigM.append(numpy.trace(CinvsigM@ C_Om@Cinv@C_Om+Cinv@ C_Om@CinvsigM@C_Om))

        ll_ind.append(C[0][0]**(-2)*C_l[0][0]**2+C[1][1]**(-2)*C_l[1][1]**2)
        bb_ind.append(C[0][0]**(-2)*C_b[0][0]**2+C[1][1]**(-2)*C_b[1][1]**2)
        lb_ind.append(C[0][0]**(-2)*C_l[0][0]*C_b[0][0]+C[1][1]**(-2)*C_l[1][1]*C_b[1][1])
        lO_ind.append(C[0][0]**(-2)*C_l[0][0]*C_Om[0][0]+C[1][1]**(-2)*C_l[1][1]*C_Om[1][1])
        bO_ind.append(C[0][0]**(-2)*C_b[0][0]*C_Om[0][0]+C[1][1]**(-2)*C_b[1][1]*C_Om[1][1])
        OO_ind.append(C[0][0]**(-2)*C_Om[0][0]**2+C[1][1]**(-2)*C_Om[1][1]**2)

        ll_vonly.append(C[1][1]**(-2)*C_l[1][1]**2)
        lO_vonly.append(C[1][1]**(-2)*C_l[1][1]*C_Om[1][1])
        OO_vonly.append(C[1][1]**(-2)*C_Om[1][1]**2)

    return numpy.array(ll),numpy.array(bb),numpy.array(lb),numpy.array(lls),numpy.array(bbs),numpy.array(lbs), \
        numpy.array(ll_ind),numpy.array(bb_ind),numpy.array(lb_ind),numpy.array(llsigM),numpy.array(bbsigM),numpy.array(lbsigM), \
        numpy.array(ll_vonly),numpy.array(lO),numpy.array(bO),numpy.array(OO),numpy.array(lOs),numpy.array(bOs),numpy.array(OOs), \
        numpy.array(lO_ind),numpy.array(bO_ind),numpy.array(OO_ind),numpy.array(lOsigM),numpy.array(bOsigM),numpy.array(OOsigM), \
        numpy.array(lO_vonly), numpy.array(OO_vonly)

def traces_fast(z,mu,ng,duration,sigm,restrate):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm,restrate)

    ll=[]
    lb=[]
    bb=[]
    lO=[]
    bO=[]
    OO=[]

    for C, Cinv, C_l,C_b,Cinvn,CinvsigM, C_Om in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4],cmatrices[5],cmatrices[6]):
        ll.append(numpy.trace(Cinv@C_l@Cinv@C_l))
        bb.append(numpy.trace(Cinv@C_b@Cinv@C_b))
        lb.append(numpy.trace(Cinv@C_l@Cinv@C_b))
        lO.append(numpy.trace(Cinv@C_l@Cinv@C_Om))
        bO.append(numpy.trace(Cinv@C_b@Cinv@C_Om))
        OO.append(numpy.trace(Cinv@C_Om@Cinv@C_Om))

    return numpy.array(ll),numpy.array(bb),numpy.array(lb),numpy.array(lO),numpy.array(bO),numpy.array(OO)

def muintegral(z,ng,duration,sigm,restrate):
    mus=numpy.arange(0,1.001,0.05)

    ll=numpy.zeros((len(mus),len(matter[:,0])))
    bb=numpy.zeros((len(mus),len(matter[:,0])))
    bl=numpy.zeros((len(mus),len(matter[:,0])))
    lO=numpy.zeros((len(mus),len(matter[:,0])))
    bO=numpy.zeros((len(mus),len(matter[:,0])))
    OO=numpy.zeros((len(mus),len(matter[:,0])))
    lls=numpy.zeros((len(mus),len(matter[:,0])))
    bbs=numpy.zeros((len(mus),len(matter[:,0])))
    bls=numpy.zeros((len(mus),len(matter[:,0])))
    lOs=numpy.zeros((len(mus),len(matter[:,0])))
    bOs=numpy.zeros((len(mus),len(matter[:,0])))
    OOs=numpy.zeros((len(mus),len(matter[:,0])))
    ll_ind=numpy.zeros((len(mus),len(matter[:,0])))
    bb_ind=numpy.zeros((len(mus),len(matter[:,0])))
    bl_ind=numpy.zeros((len(mus),len(matter[:,0])))
    lO_ind=numpy.zeros((len(mus),len(matter[:,0])))
    bO_ind=numpy.zeros((len(mus),len(matter[:,0])))
    OO_ind=numpy.zeros((len(mus),len(matter[:,0])))    
    llsigM=numpy.zeros((len(mus),len(matter[:,0])))
    bbsigM=numpy.zeros((len(mus),len(matter[:,0])))
    blsigM=numpy.zeros((len(mus),len(matter[:,0])))
    lOsigM=numpy.zeros((len(mus),len(matter[:,0])))
    bOsigM=numpy.zeros((len(mus),len(matter[:,0])))
    OOsigM=numpy.zeros((len(mus),len(matter[:,0])))
    ll_vonly=numpy.zeros((len(mus),len(matter[:,0])))  
    lO_vonly=numpy.zeros((len(mus),len(matter[:,0])))  
    OO_vonly=numpy.zeros((len(mus),len(matter[:,0])))  


    for i in range(len(mus)):
        dum = traces(z,mus[i],ng,duration,sigm,restrate)
        ll[i],bb[i],bl[i],lls[i],bbs[i],bls[i],ll_ind[i],bb_ind[i],bl_ind[i],llsigM[i],bbsigM[i],blsigM[i],ll_vonly[i],lO[i],bO[i],OO[i], \
            lOs[i],bOs[i],OOs[i],lO_ind[i],bO_ind[i],OO_ind[i],lOsigM[i],bOsigM[i],OOsigM[i],lO_vonly[i],OO_vonly[i] = traces(z,mus[i],ng,duration,sigm,restrate)

    ll = numpy.trapz(ll,mus,axis=0)
    bb = numpy.trapz(bb,mus,axis=0)
    bl = numpy.trapz(bl,mus,axis=0)
    lO = numpy.trapz(lO,mus,axis=0)
    bO = numpy.trapz(bO,mus,axis=0)
    OO = numpy.trapz(OO,mus,axis=0)

    lls = numpy.trapz(lls,mus,axis=0)
    bbs = numpy.trapz(bbs,mus,axis=0)
    bls = numpy.trapz(bls,mus,axis=0)
    lOs = numpy.trapz(lOs,mus,axis=0)
    bOs = numpy.trapz(bOs,mus,axis=0)
    OOs = numpy.trapz(OOs,mus,axis=0)
    ll_ind = numpy.trapz(ll_ind,mus,axis=0)
    bb_ind = numpy.trapz(bb_ind,mus,axis=0)
    bl_ind = numpy.trapz(bl_ind,mus,axis=0)
    lO_ind = numpy.trapz(lO_ind,mus,axis=0)
    bO_ind = numpy.trapz(bO_ind,mus,axis=0)
    OO_ind = numpy.trapz(OO_ind,mus,axis=0)
    llsigM = numpy.trapz(llsigM,mus,axis=0)
    bbsigM = numpy.trapz(bbsigM,mus,axis=0)
    blsigM = numpy.trapz(blsigM,mus,axis=0)
    lOsigM = numpy.trapz(lOsigM,mus,axis=0)
    bOsigM = numpy.trapz(bOsigM,mus,axis=0)
    OOsigM = numpy.trapz(OOsigM,mus,axis=0)
    ll_vonly = numpy.trapz(ll_vonly,mus,axis=0)
    lO_vonly = numpy.trapz(lO_vonly,mus,axis=0)
    OO_vonly = numpy.trapz(OO_vonly,mus,axis=0)
    return 2*ll,2*bb,2*bl, 2*lls,2*bbs,2*bls,2*ll_ind,2*bb_ind,2*bl_ind, 2*llsigM,2*bbsigM,2*blsigM,2*ll_vonly, 2*lO, 2*bO, 2*OO, \
        2*lOs,2*bOs,2*OOs,2*lO_ind,2*bO_ind,2*OO_ind, 2*lOsigM,2*bOsigM,2*OOsigM, 2*lO_vonly,2*OO_vonly

def muintegral_fast(z,ng,duration,sigm,restrate):
    mus=numpy.arange(0,1.001,0.05)

    ll=numpy.zeros((len(mus),len(matter[:,0])))
    bb=numpy.zeros((len(mus),len(matter[:,0])))
    bl=numpy.zeros((len(mus),len(matter[:,0])))
    lO=numpy.zeros((len(mus),len(matter[:,0])))
    bO=numpy.zeros((len(mus),len(matter[:,0])))
    OO=numpy.zeros((len(mus),len(matter[:,0])))

    for i in range(len(mus)):
        # dum = traces_fast(z,mus[i],ng,duration,sigm,restrate)
        ll[i],bb[i],bl[i], lO[i],bO[i],OO[i]= traces_fast(z,mus[i],ng,duration,sigm,restrate)

    ll = numpy.trapz(ll,mus,axis=0)
    bb = numpy.trapz(bb,mus,axis=0)
    bl = numpy.trapz(bl,mus,axis=0)
    lO = numpy.trapz(lO,mus,axis=0)
    bO = numpy.trapz(bO,mus,axis=0)
    OO = numpy.trapz(OO,mus,axis=0)

    return 2*ll,2*bb,2*bl, 2*lO, 2*bO, 2*OO


def kintegral(z,zmax,ng,duration,sigm,restrate):
    kmin = numpy.pi/(zmax*3e3)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)

    ll,bb,bl, lls,bbs,bls,ll_ind,bb_ind,bl_ind, llsigM,bbsigM,blsigM,ll_vonly,lO,bO,OO,\
        lOs,bOs,OOs,lO_ind,bO_ind,OO_ind, lOsigM,bOsigM,OOsigM,lO_vonly,OO_vonly= muintegral(z,ng,duration,sigm,restrate)
    ll = numpy.trapz(matter[:,0][w]**2*ll[w],matter[:,0][w])
    bb = numpy.trapz(matter[:,0][w]**2*bb[w],matter[:,0][w])
    bl = numpy.trapz(matter[:,0][w]**2*bl[w],matter[:,0][w])
    lO = numpy.trapz(matter[:,0][w]**2*lO[w],matter[:,0][w])
    bO = numpy.trapz(matter[:,0][w]**2*bO[w],matter[:,0][w])
    OO = numpy.trapz(matter[:,0][w]**2*OO[w],matter[:,0][w])
    lls = numpy.trapz(matter[:,0][w]**2*lls[w],matter[:,0][w])
    bbs = numpy.trapz(matter[:,0][w]**2*bbs[w],matter[:,0][w])
    bls = numpy.trapz(matter[:,0][w]**2*bls[w],matter[:,0][w])
    lOs = numpy.trapz(matter[:,0][w]**2*lOs[w],matter[:,0][w])
    bOs = numpy.trapz(matter[:,0][w]**2*bOs[w],matter[:,0][w])
    OOs = numpy.trapz(matter[:,0][w]**2*OOs[w],matter[:,0][w])
    ll_ind = numpy.trapz(matter[:,0][w]**2*ll_ind[w],matter[:,0][w])
    bb_ind = numpy.trapz(matter[:,0][w]**2*bb_ind[w],matter[:,0][w])
    bl_ind = numpy.trapz(matter[:,0][w]**2*bl_ind[w],matter[:,0][w])
    lO_ind = numpy.trapz(matter[:,0][w]**2*lO_ind[w],matter[:,0][w])
    bO_ind = numpy.trapz(matter[:,0][w]**2*bO_ind[w],matter[:,0][w])
    OO_ind = numpy.trapz(matter[:,0][w]**2*OO_ind[w],matter[:,0][w])    
    llsigM = numpy.trapz(matter[:,0][w]**2*llsigM[w],matter[:,0][w])
    bbsigM = numpy.trapz(matter[:,0][w]**2*bbsigM[w],matter[:,0][w])
    blsigM = numpy.trapz(matter[:,0][w]**2*blsigM[w],matter[:,0][w])
    lOsigM = numpy.trapz(matter[:,0][w]**2*lOsigM[w],matter[:,0][w])
    bOsigM = numpy.trapz(matter[:,0][w]**2*bOsigM[w],matter[:,0][w])
    OOsigM = numpy.trapz(matter[:,0][w]**2*OOsigM[w],matter[:,0][w])
    ll_vonly = numpy.trapz(matter[:,0][w]**2*ll_vonly[w],matter[:,0][w])
    lO_vonly = numpy.trapz(matter[:,0][w]**2*lO_vonly[w],matter[:,0][w])
    OO_vonly = numpy.trapz(matter[:,0][w]**2*OO_vonly[w],matter[:,0][w])
    # llkmax = numpy.interp(kmax,matter[:,0], matter[:,0]**2*ll)
    # bbkmax = numpy.interp(kmax,matter[:,0], matter[:,0]**2*bb)
    # blkmax = numpy.interp(kmax,matter[:,0], matter[:,0]**2*bl)

    return ll,bb,bl, lls,bbs,bls,ll_ind,bb_ind,bl_ind, llsigM,bbsigM,blsigM,ll_vonly,lO,bO,OO, \
        lOs,bOs,OOs,lO_ind,bO_ind,OO_ind, lOsigM,bOsigM,OOsigM,lO_vonly, OO_vonly

def kintegral_fast(z,zmax,ng,duration,sigm,restrate):
    kmin = numpy.pi/(zmax*3e3)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)

    ll,bb,bl,lO,bO,OO = muintegral_fast(z,ng,duration,sigm,restrate)
    ll = numpy.trapz(matter[:,0][w]**2*ll[w],matter[:,0][w])
    bb = numpy.trapz(matter[:,0][w]**2*bb[w],matter[:,0][w])
    bl = numpy.trapz(matter[:,0][w]**2*bl[w],matter[:,0][w])
    lO = numpy.trapz(matter[:,0][w]**2*lO[w],matter[:,0][w])
    bO = numpy.trapz(matter[:,0][w]**2*bO[w],matter[:,0][w])
    OO = numpy.trapz(matter[:,0][w]**2*OO[w],matter[:,0][w])

    return ll,bb,bl,lO,bO,OO 

# utility numbers
zmax_zint = 0.3
zs_zint = numpy.arange(0.01,0.3+0.00001,0.01) # in redshift space
rs_zint = cosmo.comoving_distance(zs_zint).value

def zintegral(zmax,ng,duration,sigm,restrate):
    # zs = numpy.arange(0.01,zmax+0.00001,0.01) # in redshift space


    # zs = zs*3e5/100
    # dz = 0.01*3e5/100

    w = zs_zint <= zmax
    zs = zs_zint[w]
    rs = rs_zint[w]

    ll=numpy.zeros(len(rs))
    bb=numpy.zeros(len(rs))
    bl=numpy.zeros(len(rs))
    lO=numpy.zeros(len(rs))
    bO=numpy.zeros(len(rs))
    OO=numpy.zeros(len(rs))
    lls=numpy.zeros(len(rs))
    bbs=numpy.zeros(len(rs))
    bls=numpy.zeros(len(rs))
    lOs=numpy.zeros(len(rs))
    bOs=numpy.zeros(len(rs))
    OOs=numpy.zeros(len(rs))
    ll_ind=numpy.zeros(len(rs))
    bb_ind=numpy.zeros(len(rs))
    bl_ind=numpy.zeros(len(rs))
    lO_ind=numpy.zeros(len(rs))
    bO_ind=numpy.zeros(len(rs))
    OO_ind=numpy.zeros(len(rs))
    llsigM=numpy.zeros(len(rs))
    bbsigM=numpy.zeros(len(rs))
    blsigM=numpy.zeros(len(rs))
    lOsigM=numpy.zeros(len(rs))
    bOsigM=numpy.zeros(len(rs))
    OOsigM=numpy.zeros(len(rs))
    ll_vonly=numpy.zeros(len(rs))
    lO_vonly=numpy.zeros(len(rs))
    OO_vonly=numpy.zeros(len(rs))
        # llkmax=numpy.zeros(len(rs))
    # bbkmax=numpy.zeros(len(rs))
    # blkmax=numpy.zeros(len(rs))

    for i in range(len(rs)):
        ll[i],bb[i],bl[i],lls[i],bbs[i],bls[i],ll_ind[i],bb_ind[i],bl_ind[i],llsigM[i],bbsigM[i],blsigM[i],ll_vonly[i],lO[i],bO[i],OO[i], \
            lOs[i],bOs[i],OOs[i],lO_ind[i],bO_ind[i],OO_ind[i],lOsigM[i],bOsigM[i],OOsigM[i],lO_vonly[i],OO_vonly[i]  = kintegral(zs[i],zmax,ng,duration,sigm,restrate)

    ll = numpy.trapz(rs**2*ll,rs)
    bb = numpy.trapz(rs**2*bb,rs)
    bl = numpy.trapz(rs**2*bl,rs)
    lO = numpy.trapz(rs**2*lO,rs)
    bO = numpy.trapz(rs**2*bO,rs)
    OO = numpy.trapz(rs**2*OO,rs)
    lls = numpy.trapz(rs**2*lls,rs)
    bbs = numpy.trapz(rs**2*bbs,rs)
    bls = numpy.trapz(rs**2*bls,rs)
    lOs = numpy.trapz(rs**2*lOs,rs)
    bOs = numpy.trapz(rs**2*bOs,rs)
    OOs = numpy.trapz(rs**2*OOs,rs)    
    ll_ind = numpy.trapz(rs**2*ll_ind,rs)
    bb_ind = numpy.trapz(rs**2*bb_ind,rs)
    bl_ind = numpy.trapz(rs**2*bl_ind,rs)
    lO_ind = numpy.trapz(rs**2*lO_ind,rs)
    bO_ind = numpy.trapz(rs**2*bO_ind,rs)
    OO_ind = numpy.trapz(rs**2*OO_ind,rs)
    llsigM = numpy.trapz(rs**2*llsigM,rs)
    bbsigM = numpy.trapz(rs**2*bbsigM,rs)
    blsigM = numpy.trapz(rs**2*blsigM,rs)
    lOsigM = numpy.trapz(rs**2*lOsigM,rs)
    bOsigM = numpy.trapz(rs**2*bOsigM,rs)
    OOsigM = numpy.trapz(rs**2*OOsigM,rs)
    ll_vonly = numpy.trapz(rs**2*ll_vonly,rs)
    lO_vonly = numpy.trapz(rs**2*lO_vonly,rs)
    OO_vonly = numpy.trapz(rs**2*OO_vonly,rs)
            # llkmax = numpy.trapz(rs**2*llkmax,rs)
    # bbkmax = numpy.trapz(rs**2*bbkmax,rs)
    # blkmax = numpy.trapz(rs**2*blkmax,rs)

    return ll, bb, bl, lls,bbs,bls,ll_ind, bb_ind, bl_ind, llsigM,bbsigM,blsigM,ll_vonly,lO,bO,OO, \
        lOs,bOs,OOs,lO_ind, bO_ind, OO_ind, lOsigM,bOsigM,OOsigM,lO_vonly,OO_vonly


def zintegral_fast(zmax,ng,duration,sigm,restrate):
    w = zs_zint <= zmax
    zs = zs_zint[w]
    rs = rs_zint[w]

    ll=numpy.zeros(len(rs))
    bb=numpy.zeros(len(rs))
    bl=numpy.zeros(len(rs))
    lO=numpy.zeros(len(rs))
    bO=numpy.zeros(len(rs))
    OO=numpy.zeros(len(rs))

    for i in range(len(rs)):
        ll[i],bb[i],bl[i],lO[i],bO[i],OO[i] = kintegral_fast(zs[i],zmax,ng,duration,sigm,restrate)

    ll = numpy.trapz(rs**2*ll,rs)
    bb = numpy.trapz(rs**2*bb,rs)
    bl = numpy.trapz(rs**2*bl,rs)
    lO = numpy.trapz(rs**2*lO,rs)
    bO = numpy.trapz(rs**2*bO,rs)
    OO = numpy.trapz(rs**2*OO,rs)
    return ll, bb, bl, lO,bO,OO


def set2():
    fig,(ax) = plt.subplots(1, 1)
    durations = [2.,10.]
    labels = ['Two Years','Ten Years']
    colors = ['red','black']

    zmaxs=[0.05,0.2]
    lss = [':','-']
    zmaxs=[0.2]
    lss = ['-']

    for ls,zmax in zip(lss, zmaxs):
        zs = numpy.arange(0.01,zmax+0.00001,0.01)
        rs = cosmo.comoving_distance(zs).value


        for duration,label,color in zip(durations,labels,colors):
            dvardz=[]
            f00,f11,f10, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM,f00_vonly,f02,f12,f22, \
                f02s,f12s,f22s,f02_ind,f12_ind,f22_ind, f02sigM,f12sigM,f22sigM, f01_vonly, f11_vonly = zintegral(zmax,ng,duration,sigm_Ia,restrate_Ia)
            var= numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0]*2*3.14/.75
            for z,r in zip(zs,rs):
                a=1./(1+z)
                drdz = 1/numpy.sqrt(OmegaM0/a**3 + (1-OmegaM0)) # 1/H
                _,_,_, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM,f00_vonly,f02,f12,f22, \
                    f02s,f12s,f22s,f02_ind,f12_ind,f22_ind, f02sigM,f12sigM,f22sigM, f01_vonly, f11_vonly= kintegral(z,zmax,ng,duration,sigm_Ia,restrate_Ia)
                dvardz.append(finvp3d(f00,f10,f02,f11,f12,f22+sigOM0sqinv, f00s,f10s,f02s,f11s,f12s,f22s))
                dvardz[-1] = dvardz[-1]/r**2
                dvardz[-1] = dvardz[-1]*drdz  # z now has units of 100 km/s
                dvardz[-1] = dvardz[-1]* 3e3
            dvardz=numpy.array(dvardz)*2*3.14/.75
            plt.plot(zs,-dvardz/2/numpy.sqrt(var),label=r'{} $z_{{max}}={}$'.format(label,zmax),color=color,ls=ls)
    plt.xlabel('z')
    plt.yscale("log", nonposy='clip')
    plt.ylabel(r'$\sigma_{\gamma}^{-1}|\frac{d\sigma_{\gamma}}{dz}|$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dvardz.png')
    plt.clf()

# set2()
# wefwe

def set1():
    fig,(ax) = plt.subplots(1, 1)
    zmaxs = numpy.exp(numpy.arange(numpy.log(0.05),numpy.log(.300001),numpy.log(.3/.05)/8))
    durations = [2.,10.]
    labels = ['Two Years','Ten Years']
    var =[]
    dvards = []
    var_ind=[]
    dvardsigM = []
    var_vonly=[]
    dvardkmax = []
    for duration,label in zip(durations,labels):
        v_=[]
        vind_=[]
        vvonly_=[]
        dv_=[]
        dvsigM_=[]
        dvdkmax_=[]
        for zmax in zmaxs:
            f00,f11,f10, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM,f00_vonly,f02,f12,f22, \
                f02s,f12s,f22s,f02_ind,f12_ind,f22_ind, f02sigM,f12sigM,f22sigM, f01_vonly, f11_vonly = zintegral(zmax,ng,duration,sigm_Ia,restrate_Ia)
            # dv_.append(finvp(f00,f11,f10,f00s,f11s,f10s))
            dv_.append(finvp3d(f00,f10,f02,f11,f12,f22+sigOM0sqinv, f00s,f10s,f02s,f11s,f12s,f22s))
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0])
            vind_.append(numpy.linalg.inv(numpy.array([[f00_ind,f10_ind,f02_ind],[f10_ind,f11_ind,f12_ind],[f02_ind,f12_ind,f22_ind+sigOM0sqinv]]))[0,0])
            vvonly_.append(numpy.linalg.inv(numpy.array([[f00_vonly,f01_vonly],[f01_vonly,f11_vonly+sigOM0sqinv]]))[0,0])
            dvsigM_.append(finvp3d(f00,f10,f02,f11,f12,f22+sigOM0sqinv,f00sigM,f10sigM,f02sigM,f11sigM,f12sigM,f22sigM))
            # dvdkmax_.append(finvp(f00,f11,f10,f00kmax,f11kmax,f10kmax))


        var.append(numpy.array(v_)*2*3.14/.75) #3/4
        dvards.append(numpy.array(dv_)*2*3.14/.75)
        var_ind.append(numpy.array(vind_)*2*3.14/.75) #3/4
        dvardsigM.append(numpy.array(dvsigM_)*2*3.14/.75)
        var_vonly.append(numpy.array(vvonly_)*2*3.14/.75) #3/4   
        # dvardkmax.append(numpy.array(dvdkmax_)*2*3.14/.75)
 
    plt.plot(zmaxs,numpy.sqrt(var_vonly[0]),label='Two Years, Velocity Only',color='red',ls='--')  
    plt.plot(zmaxs,numpy.sqrt(var_ind[0]),label='Two Years, Independent Surveys',color='red',ls=':')  
    plt.plot(zmaxs,numpy.sqrt(var[0]),label='Two Years, Cross-Correlated',color='red')
    plt.plot(zmaxs,numpy.sqrt(var_vonly[1]),label='Ten Years, Velocity Only',color='black',ls='--') 
    plt.plot(zmaxs,numpy.sqrt(var_ind[1]),label='Ten Years, Independent Surveys',color='black',ls=':')   
    plt.plot(zmaxs,numpy.sqrt(var[1]),label='Ten Years, Cross-Correlated',color='black')

    plt.xlabel(r'$z_{max}$')
    plt.ylim((0,0.07))
    plt.ylabel(r'$\sigma_{\gamma}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('var.png')

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(zmaxs,-dvards[0]/2/var[0],label='Two Years',color='red',ls='--')
    ax1.plot(zmaxs,-dvards[1]/2/var[1],label='Ten Years',color='red')
    # ax1.yscale("log", nonposy='clip')
    ax1.set_ylim((0,ax1.get_ylim()[1]))
    ax1.set_xlabel(r'$z_{max}$')
    ax1.set_ylabel(r'$\sigma_{\gamma}^{-1}|\frac{d\sigma_{\gamma}}{d\ln{t}}|$',color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(zmaxs,dvardsigM[0]/2/var[0],label='Two Years',color='black',ls='--')
    ax2.plot(zmaxs,dvardsigM[1]/2/var[1],label='Ten Years',color='black')

    # ax2.yscale("log", nonposy='clip')
    ax2.set_ylim((0,ax2.get_ylim()[1]))
    ax2.set_ylabel(r'$\sigma_{\gamma}^{-1}\frac{d\sigma_{\gamma}}{d\sigma_M}$')
    fig.tight_layout()
    plt.legend()
    plt.savefig('dvardxxx.png')
    plt.clf()

#set1()

def snIIp():
    fig,(ax) = plt.subplots(1, 1)
    zmax = 0.05
    sigs=[0.1,0.2]
    restrate=4.5*restrate_Ia
    durations = [2,10]
    labels = ['Two Years','Ten Years']
    var =[]
    dvards = []
    var_ind=[]
    dvardsigM = []
    var_vonly=[]
    dvardkmax = []
    for duration,label in zip(durations,labels):
        v_=[]
        vind_=[]
        vvonly_=[]
        dv_=[]
        dvsigM_=[]
        dvdkmax_=[]
        for sigm in sigs:
            f00,f11,f10, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM,f00_vonly,f02,f12,f22, \
                f02s,f12s,f22s,f02_ind,f12_ind,f22_ind, f02sigM,f12sigM,f22sigM, f01_vonly, f11_vonly = zintegral(zmax,ng,duration,sigm_Ia,restrate_Ia)
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0])

        var.append(numpy.array(v_)*2*3.14/.75) #3/4
        # dvards.append(numpy.array(dv_)*2*3.14/.75)
        # var_ind.append(numpy.array(vind_)*2*3.14/.75) #3/4
        # dvardsigM.append(numpy.array(dvsigM_)*2*3.14/.75)
        # var_vonly.append(numpy.array(vvonly_)*2*3.14/.75) #3/4   
        # dvardkmax.append(numpy.array(dvdkmax_)*2*3.14/.75)

    print (numpy.sqrt(var[0]))
    print (numpy.sqrt(var[1]))


#snIIp()



def fD_OmegaM():
    gamma=0.55
    oms = [0.28-0.013, 0.28, 0.28+0.013]
    zs = numpy.arange(0,1,0.1)
    as_ = 1/(1+zs)
    for om in oms:
        ans=[]
        for a in as_:
            ans.append(OmegaM(a,om)**0.55*D(a,om))
        plt.plot(zs,ans,label=r'$\Omega_M={}$'.format(om))
    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$fD$')
    plt.tight_layout()
    plt.show()
    plt.clf()

#fD_OmegaM()

# This is the code to determine the survey Figure of Merit
# The inputs fron Nicolas are the solid angle covered and the number of SNe (if I remember correctly)
# The inputs in this code are fraction of sky and survey duration
# Calculating skyfrac is trivial
# Calculating duration is harder
# I think that the thing to do is take Nicolas output and make a histogram of number of SNe / solid angle
# The histogram should should a distribution with a sharp cutoff.
# The x-scale of the histogram can be renormalized so that the maximum of the histogram has duration=10
# Then the following code should give an "inverse" figure of merit

def surveyFOM(skyfrac, duration):
    zmax = 0.2

    f00,f11,f10, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM,f00_vonly,f02,f12,f22, \
        f02s,f12s,f22s,f02_ind,f12_ind,f22_ind, f02sigM,f12sigM,f22sigM, f01_vonly, f11_vonly = zintegral(zmax,ng,duration,sigm_Ia,restrate_Ia)

    return numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0]*2*3.14/skyfrac


#surveyFOM()