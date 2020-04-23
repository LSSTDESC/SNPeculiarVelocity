#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker, cm
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM
import astropy
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2.0

# cosmology
OmegaM0 = 0.30
cosmo = FlatLambdaCDM(H0=100, Om0=OmegaM0)

gamma=0.55
sigma_v=300.

# tho code is hard wired to change these numbers requires surgery
zmax=0.3
nbins = 3
binwidth = zmax/nbins
bincenters = binwidth/2 + binwidth*numpy.arange(nbins)
print(bincenters)

# power spectrum from CAMB
matter = numpy.loadtxt("/Users/akim/project/PeculiarVelocity/pv/dragan/matterpower.dat")

# vv
# f=OmegaM0**0.55

# SN properties
restrate_Ia = 0.65*  2.69e-5*(1/0.70)**3 # h^3/Mpc^3 yr^{-1}
sigm_Ia = 0.08

# galaxy density (TAIPAN Howlett et al.)
ng = 1e-3 # not used in current code

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

# def dlnfs8dg(a,*args):
#     return numpy.log(OmegaM(a,*args)) + integrate.quad(integrand_dlnfs, 1e-8, a)[0]


# def dfdg(a,*args):
#     om = OmegaM(a,*args)
#     return om**gamma * numpy.log(om)

# def integrand_dDdg(a,*args):
#     return dfdg(a,*args) / a

# def dDdg(a,*args):
#     return D(a,*args) * integrate.quad(integrand_dDdg, 1e-8, a)[0]

def integrand_D(a,gamma=gamma, OmegaM0=OmegaM0,*args):
    return  OmegaM(a,OmegaM0,*args)**gamma / a

# normalized to a=1
def D(a,OmegaM0=OmegaM0, gamma=gamma, *args):
    #return 1./1000*numpy.exp(integrate.quad(integrand_D, 1./1000,a,args=(gamma,OmegaM0))[0])
    return numpy.exp(-integrate.quad(integrand_D,a,1,args=(gamma,OmegaM0))[0])

# def dOmdOm0(a,OmegaM0=OmegaM0):
#     den= (OmegaM0 + (1-OmegaM0)*a**3) 
#     return (1./den - (1-a**3)/den**2)

# def integrand_dDdOmOverD(a,OmegaM0=OmegaM0, gamma=gamma,*args):
#     return  gamma*OmegaM(a,*args)**(gamma-1) / a

# def dDdOmOverD(a, *args):
#     return integrate.quad(integrand_dDdOmOverD, a,1,args=args)[0]



def Pvv(mu,fD):
    return (fD*mu*100)**2*matter[:,1]/matter[:,0]**2

def Pvv_fD(mu,fD):
    return 2*mu*100*(fD*mu*100)*matter[:,1]/matter[:,0]**2

# gg
b= 1.2
def Pgg(mu,fD,bD):
    return (bD+fD*mu**2)**2*matter[:,1]


def Pgg_b(mu,fD,bD):
#    return 2*(b*D+f*D*mu**2)*matter[:,1]   
    return 2*(bD+fD*mu**2)*matter[:,1]

def Pgg_fD(mu,fD,bD):
    return 2*(mu**2)*(bD+fD*mu**2)*matter[:,1]

# gv
def Pgv(mu,fD,bD):
    return (bD+fD*mu**2)*(fD*mu*100)*matter[:,1]/matter[:,0]

def Pgv_b(mu,fD,bD):
    # return (f*D*mu*100)*matter[:,1]/matter[:,0]
    return (fD*mu*100)*matter[:,1]/matter[:,0]

def Pgv_fD(mu,fD,bD):
    return ((mu**2)*(fD*mu*100)  +(bD+fD*mu**2)*(mu*100))      *matter[:,1]/matter[:,0]


def finvp(f00,f11,f01,f00p,f11p,f01p):
    den = f00*f11 -f01**2
    return f11p /den - f11*(f00p*f11 + f00*f11p - 2*f01*f01p)/den**2

def Cmatrices(z,mu,ng,duration,sigm,restrate):
    zbin = numpy.argmin(abs(bincenters-z))
    a = 1/(1.+z)
    # OmegaM_a = OmegaM(a)
    n = duration*restrate/(1+z)
    # sigv_factor = numpy.log(10)/5*z/(1+z)
    # sigv =sigm * sigv_factor *3e5
    # print(sigv_factor,sigv)
    effz = (cosmo.H(z)*cosmo.comoving_distance(z)/astropy.constants.c).decompose().value
    # print('effx',effz)
    # print(z,cosmo.H(z),cosmo.comoving_distance(z),cosmo.H(z)*cosmo.comoving_distance(z)/astropy.constants.c)
    term = 5/numpy.log(10)*numpy.abs(1-1/effz)
    # print (term)
    # print (1/term)
    sigv_factor=1/term
    sigv = 3e5*sigm*sigv_factor
    sigv = numpy.sqrt(sigv**2+sigma_v**2)

    # lnfgfactor = dlnfs8dg(a)

    sigv2 = sigv**2
    nvinv = sigv2/n
    ninv = 1./n

    # nginv = 1./ng

    # f = OmegaM(a)**.55
    # D_ = D(a)
    # dDdg_ = dDdg(a)
    # dfdg_ = dfdg(a)
    # dfdOm_ = dfdOm(a)
    # dDdOmOverD_ = dDdOmOverD(a)

    fD_z = fD(z)
    bD_z = bD(z)

    pggs = Pgg(mu,fD_z,bD_z)
    pgvs = Pgv(mu,fD_z,bD_z)
    pvvs = Pvv(mu,fD_z)

    pggs_fD = Pgg_fD(mu,fD_z,bD_z)
    pgvs_fD = Pgv_fD(mu,fD_z,bD_z)
    pvvs_fD = Pvv_fD(mu,fD_z)

    pggs_b = Pgg_b(mu,fD_z,bD_z)
    pgvs_b = Pgv_b(mu,fD_z,bD_z)


    C=[]
    Cinv = []
    dCdX1 = []
    dCdX2 = []
    dCdX3 = []
    dCdb = []
    # Cinvs = []
    Cinvn = []
    Cinvsigmam = []
    for pgg, pgv, pvv, pgg_fD, pgv_fD, pvv_fD,  pgg_b, pgv_b  in zip(pggs,pgvs,pvvs,pggs_fD, pgvs_fD, pvvs_fD,pggs_b,pgvs_b):
        C.append(numpy.array([[pgg+ninv,pgv],[pgv,pvv+nvinv]]))
        Cinv.append(numpy.linalg.inv(C[-1]))
        if z> 0 and z<= 0.1:
            dCdX1.append([[pgg_fD, pgv_fD],[pgv_fD, pvv_fD]])
            dCdX2.append([[0,0],[0,0]])
            dCdX3.append([[0,0],[0,0]])
        elif z>0.1 and z<=0.2:
            dCdX1.append([[0,0],[0,0]])
            dCdX2.append([[pgg_fD, pgv_fD],[pgv_fD, pvv_fD]])
            dCdX3.append([[0,0],[0,0]])
        elif z>0.2 and z<=0.3:
            dCdX1.append([[0,0],[0,0]])
            dCdX2.append([[0,0],[0,0]])
            dCdX3.append([[pgg_fD, pgv_fD],[pgv_fD, pvv_fD]])
        else:
            print("integrating at ",z)
            # raise Exception('bad redshift')

        dCdb.append(numpy.array([[pgg_b,pgv_b],[pgv_b,0]]))

    return C, Cinv, dCdX1,dCdX2,dCdX3,dCdb



def traces_fast(z,mu,ng,duration,sigm,restrate):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm,restrate)

    X1X1=[]
    X1X2=[]
    X1X3=[]
    X2X2=[]
    X2X3=[]
    X3X3=[]
    X1O=[]
    X2O=[]
    X3O=[]
    OO=[]

    for C, Cinv, C_X1,C_X2,C_X3, C_b in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4],cmatrices[5]):
        X1X1.append(numpy.trace(Cinv@C_X1@Cinv@C_X1))
        X1X2.append(numpy.trace(Cinv@C_X1@Cinv@C_X2))
        X1X3.append(numpy.trace(Cinv@C_X1@Cinv@C_X3))
        X2X2.append(numpy.trace(Cinv@C_X2@Cinv@C_X2))
        X2X3.append(numpy.trace(Cinv@C_X2@Cinv@C_X3))
        X3X3.append(numpy.trace(Cinv@C_X3@Cinv@C_X3))
        X1O.append(numpy.trace(Cinv@C_X1@Cinv@C_b))
        X2O.append(numpy.trace(Cinv@C_X2@Cinv@C_b))
        X3O.append(numpy.trace(Cinv@C_X3@Cinv@C_b))
        OO.append(numpy.trace(Cinv@C_b@Cinv@C_b))

    return numpy.array(X1X1),numpy.array(X1X2),numpy.array(X1X3),numpy.array(X1O),numpy.array(X2X2),numpy.array(X2X3),numpy.array(X2O),numpy.array(X3X3),numpy.array(X3O),numpy.array(OO)

def muintegral_fast(z,ng,duration,sigm,restrate):
    mus=numpy.arange(0,1.001,0.05)

    x1x1=numpy.zeros((len(mus),len(matter[:,0])))
    x1x2=numpy.zeros((len(mus),len(matter[:,0])))
    x1x3=numpy.zeros((len(mus),len(matter[:,0])))
    x1O=numpy.zeros((len(mus),len(matter[:,0])))
    x2x2=numpy.zeros((len(mus),len(matter[:,0])))
    x2x3=numpy.zeros((len(mus),len(matter[:,0])))
    x2O=numpy.zeros((len(mus),len(matter[:,0])))
    x3x3=numpy.zeros((len(mus),len(matter[:,0])))
    x3O=numpy.zeros((len(mus),len(matter[:,0])))
    OO=numpy.zeros((len(mus),len(matter[:,0])))

    for i in range(len(mus)):
        # dum = traces_fast(z,mus[i],ng,duration,sigm,restrate)
        x1x1[i],x1x2[i],x1x3[i],x1O[i],x2x2[i],x2x3[i],x2O[i],x3x3[i],x3O[i],OO[i]= traces_fast(z,mus[i],ng,duration,sigm,restrate)

    x1x1=numpy.trapz(x1x1,mus,axis=0)
    x1x2=numpy.trapz(x1x2,mus,axis=0)
    x1x3=numpy.trapz(x1x3,mus,axis=0)
    x1O=numpy.trapz(x1O,mus,axis=0)
    x2x2=numpy.trapz(x2x2,mus,axis=0)
    x2x3=numpy.trapz(x2x3,mus,axis=0)
    x2O=numpy.trapz(x2O,mus,axis=0)
    x3x3=numpy.trapz(x3x3,mus,axis=0)
    x3O=numpy.trapz(x3O,mus,axis=0)
    OO=numpy.trapz(OO,mus,axis=0)


    return 2*x1x1,2*x1x2,2*x1x3,2*x1O,2*x2x2,2*x2x3,2*x2O,2*x3x3,2*x3O,2*OO


def kintegral_fast(z,zmax,ng,duration,sigm,restrate):
    kmin = numpy.pi/(zmax*3e3)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)

    x1x1,x1x2,x1x3,x1O,x2x2,x2x3,x2O,x3x3,x3O,OO = muintegral_fast(z,ng,duration,sigm,restrate)
    x1x1 = numpy.trapz(matter[:,0][w]**2*x1x1[w],matter[:,0][w])
    x1x2 = numpy.trapz(matter[:,0][w]**2*x1x2[w],matter[:,0][w])
    x1x3 = numpy.trapz(matter[:,0][w]**2*x1x3[w],matter[:,0][w])
    x1O = numpy.trapz(matter[:,0][w]**2*x1O[w],matter[:,0][w])
    x2x2 = numpy.trapz(matter[:,0][w]**2*x2x2[w],matter[:,0][w])
    x2x3 = numpy.trapz(matter[:,0][w]**2*x2x3[w],matter[:,0][w])
    x2O = numpy.trapz(matter[:,0][w]**2*x2O[w],matter[:,0][w])
    x3x3 = numpy.trapz(matter[:,0][w]**2*x3x3[w],matter[:,0][w])
    x3O = numpy.trapz(matter[:,0][w]**2*x3O[w],matter[:,0][w])
    OO = numpy.trapz(matter[:,0][w]**2*OO[w],matter[:,0][w])

    return x1x1,x1x2,x1x3,x1O,x2x2,x2x3,x2O,x3x3,x3O,OO

# utility numbers
zmax_zint = 0.3
zs_zint = numpy.arange(0.01,0.3+0.00001,0.01) # in redshift space
rs_zint = cosmo.comoving_distance(zs_zint).value




def zintegral_fast(zmax,ng,duration,sigm,restrate):
    w = zs_zint <= zmax
    zs = zs_zint[w]
    rs = rs_zint[w]

    x1x1=numpy.zeros(len(rs))
    x1x2=numpy.zeros(len(rs))
    x1x3=numpy.zeros(len(rs))
    x1O=numpy.zeros(len(rs))
    x2x2=numpy.zeros(len(rs))
    x2x3=numpy.zeros(len(rs))
    x2O=numpy.zeros(len(rs))
    x3x3=numpy.zeros(len(rs))
    x3O=numpy.zeros(len(rs))
    OO=numpy.zeros(len(rs))

    for i in range(len(rs)):
        x1x1[i],x1x2[i],x1x3[i],x1O[i],x2x2[i],x2x3[i],x2O[i],x3x3[i],x3O[i],OO[i] = kintegral_fast(zs[i],zmax,ng,duration,sigm,restrate)

    x1x1 = numpy.trapz(rs**2*x1x1,rs)
    x1x2 = numpy.trapz(rs**2*x1x2,rs)
    x1x3 = numpy.trapz(rs**2*x1x3,rs)
    x1O = numpy.trapz(rs**2*x1O,rs)
    x2x2 = numpy.trapz(rs**2*x2x2,rs)
    x2x3 = numpy.trapz(rs**2*x2x3,rs)
    x2O = numpy.trapz(rs**2*x2O,rs)
    x3x3 = numpy.trapz(rs**2*x3x3,rs)
    x3O = numpy.trapz(rs**2*x3O,rs)
    OO = numpy.trapz(rs**2*OO,rs)
    return x1x1,x1x2,x1x3,x1O,x2x2,x2x3,x2O,x3x3,x3O,OO

fD_ = []
bD_ = []

for _z in bincenters:
    a=1./(1+_z)
    f = OmegaM(a,OmegaM0=OmegaM0)**.55
    D_ = D(a,OmegaM0=OmegaM0,gamma=0.55)
    fD_.append(f*D_)
    bD_.append(1.2*D_)

fD_=numpy.array(fD_)
bD_=numpy.array(bD_)

def fD(z):
    return fD_[numpy.argmin(abs(bincenters-z))]

def bD(z):
    return bD_[numpy.argmin(abs(bincenters-z))]    


def set1():
    # fig,(ax) = plt.subplots(1, 1)
    zmaxs = [0.3]
    durations = [10.]
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
            x1x1,x1x2,x1x3,x1O,x2x2,x2x3,x2O,x3x3,x3O,OO = zintegral_fast(zmax,ng,duration,sigm_Ia,restrate_Ia)
            v_.append(numpy.linalg.inv(numpy.array([[x1x1,x1x2,x1x3,x1O],[x1x2,x2x2,x2x3,x2O],[x1x3,x2x3,x3x3,x3O],
                [x1O,x2O,x3O,OO]])))
        var.append(numpy.array(v_)*2*3.14/.5) #2/4 sky
        print(var[-1])
    print(bincenters,fD_)

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from matplotlib import ticker, cm
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def figure_fD():
    om0s = numpy.arange(0.25,0.501,0.01)
    gammas = numpy.arange(0.4,0.71,0.01)
    # om0s=[0.3]
    # gammas=[0.55]
    X, Y = numpy.meshgrid(om0s,gammas)
    zs=[0.05,.5,1]
    colors=['b','g','r','cyan']
    fig, ax = plt.subplots()
    levels=(numpy.arange(0.3,0.8,0.025))
    hs=[]
    for z,color in zip(zs,colors):
        surface=[]
        for om0 in om0s:
            surface_=[]
            for gamma in gammas:
                a=1./(1+z)
                f = OmegaM(a,OmegaM0=om0)**gamma
                D_ = D(a,OmegaM0=om0,gamma=gamma)
                surface_.append(f*D_)
                # print(om0, gamma, surface_[-1])
                # wefe
            surface.append(surface_)
        surface= numpy.flip(numpy.array(surface))
        # print(surface.shape)
        cset2 = ax.contour( surface,colors=color,extent=[X.min(), X.max(), Y.min(), Y.max()])
        h1,_ = cset2.legend_elements()
        hs.append(h1[0])
        ax.clabel(cset2, inline=True, fontsize=12)
    ax.legend(hs,[r'$z={}$'.format(z) for z in zs])
    plt.xlabel(r'$\Omega_{M0}$')
    plt.ylabel(r'$\gamma$')
    plt.tight_layout()
    plt.savefig('fDsurface.png')

def test_fD():
    om0s = numpy.arange(0.25,0.3501,0.01)
    gammas = [0.42]
    X, Y = numpy.meshgrid(om0s,gammas)
    zs=[1]
    colors=['b','g','r','cyan']
    fig, ax = plt.subplots()
    levels=(numpy.arange(0.3,0.8,0.025))
    hs=[]
    fsurface=[]
    Dsurface=[]
    fDsurface=[]
    for z,color in zip(zs,colors):
        for om0 in om0s:
            for gamma in gammas:
                a=1./(1+z)
                f = OmegaM(a,OmegaM0=om0)**gamma
                D_ = D(a,OmegaM0=om0,gamma=gamma)
                fDsurface.append(f*D_)
                fsurface.append(f)
                Dsurface.append(D_)
                # print(om0, gamma, surface_[-1])


    plt.plot(om0s,fDsurface,label='fD')
#    plt.plot(om0s,fsurface,label='f')
#    plt.plot(om0s,Dsurface,label='D')
    plt.legend()
    plt.tight_layout()
    plt.savefig('test.png')

#test_fD()
#wef
#figure_fD()


set1()