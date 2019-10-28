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
OmegaM0 = 0.28
cosmo = FlatLambdaCDM(H0=100, Om0=OmegaM0)

gamma=0.55

# tho code is hard wired to change these numbers requires surgery
zmax=0.2
nbins = 2
binwidth = zmax/nbins
bincenters = binwidth/2 + binwidth*numpy.arange(nbins)

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
    return 1./1000*numpy.exp(integrate.quad(integrand_D, 1./1000,a,args=(gamma,OmegaM0))[0])
    # return numpy.exp(-integrate.quad(integrand_D,a,1,args=(gamma,OmegaM0))[0])

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
        elif z>0.1 and z<=0.2:
            dCdX1.append([[0,0],[0,0]])
            dCdX2.append([[pgg_fD, pgv_fD],[pgv_fD, pvv_fD]])
        else:
            print("integrating at ",z)
            # raise Exception('bad redshift')

        # dCdl.append(numpy.array([[pgg_l,pgv_l],[pgv_l,pvv_l]]))   #*OmegaM_a**0.55*lnfgfactor) # second term converts from fsigma8 to gamma
        dCdb.append(numpy.array([[pgg_b,pgv_b],[pgv_b,0]]))
        # dCdOm.append(numpy.array([[pgg_Om,pgv_Om],[pgv_Om,pvv_Om]]))

        # den = (pgg+ninv)*(pvv+nvinv)-pgv**2
        # Cinvs.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+nginv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+nginv]]))
        # Cinvs[-1]=-Cinvs[-1]*sigv**2/restrate/duration**2 #in terms of lnt not s = sigma^2/rate/s

        # Cinvn.append(-1./den * numpy.array([[sigv2,0],[0,1]]) + (sigv2*(pgg+ninv)+(pvv+nvinv))/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))
        # Cinvn.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 1, sigv2,0)) # partial with respect to 1/n

        # Cinvn[-1]=Cinvn[-1]/n #in terms of lnt  n**-2 * n = 1/n

        # Cinvn.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, -ninv, -sigv2*ninv,0)) # partial wrt lnt
        # Cinvsigmam.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+ninv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))

        # print (Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 0, 2*sigv*ninv*sigv_factor *3e5 ,0))
        # Cinvsigmam[-1] = Cinvsigmam[-1] * sigv_factor *3e5 * 2 * sigv / n
        # Cinvsigmam.append(Cinverse_partial(pgg+ninv, pvv+nvinv,pgv, 0, 2*sigv*ninv*sigv_factor *3e5 ,0)) # partial wrt sigmaM

       
    return C, Cinv, dCdX1,dCdX2,dCdb



def traces_fast(z,mu,ng,duration,sigm,restrate):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm,restrate)

    ll=[]
    lb=[]
    bb=[]
    lO=[]
    bO=[]
    OO=[]

    for C, Cinv, C_l,C_b,C_Om in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4]):
        ll.append(numpy.trace(Cinv@C_l@Cinv@C_l))
        bb.append(numpy.trace(Cinv@C_b@Cinv@C_b))
        lb.append(numpy.trace(Cinv@C_l@Cinv@C_b))
        lO.append(numpy.trace(Cinv@C_l@Cinv@C_Om))
        bO.append(numpy.trace(Cinv@C_b@Cinv@C_Om))
        OO.append(numpy.trace(Cinv@C_Om@Cinv@C_Om))

    return numpy.array(ll),numpy.array(bb),numpy.array(lb),numpy.array(lO),numpy.array(bO),numpy.array(OO)

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

fD_ = []
bD_ = []

for _z in bincenters:
    a=1./(1+_z)
    f = OmegaM(a,OmegaM0=0.28)**.55
    D_ = D(a,OmegaM0=0.28,gamma=0.55)
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
    zmaxs = [0.2]
    durations = [2,10.]
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
            f00,f11,f10, f02,f12,f22 = zintegral_fast(zmax,ng,duration,sigm_Ia,restrate_Ia)
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22]])))
        var.append(numpy.array(v_)*2*3.14/.5) #2/4 sky
        print(var[-1])
    print(fD_)

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