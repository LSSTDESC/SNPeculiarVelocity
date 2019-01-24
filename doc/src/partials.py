#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM

# cosmology
OmegaM0 = 0.28
cosmo = FlatLambdaCDM(H0=100, Om0=OmegaM0)

# power spectrum from CAMB
matter = numpy.loadtxt("../../pv/dragan/matterpower.dat")

# vv
f=OmegaM0**0.55

# SN properties
restrate = 0.65*  2.69e-5*(1/0.70)**3 # h^3/Mpc^3 yr^{-1}
sigm = 0.08

# galaxy density (TAIPAN Howlett et al.)
ng = 1e-3



def OmegaM(a):
    return OmegaM0/(OmegaM0 + 0.72*a**3)

def integrand_dlnfs(a):
    return numpy.log(OmegaM(a))* OmegaM(a)**0.55 / a

def dlnfs8dg(a):
    return numpy.log(OmegaM(a)) + integrate.quad(integrand_dlnfs, 1e-8, a)[0]

def integrand_D(a):
    return  OmegaM(a)**0.55 / a

# normalized to a=1
def D(a):
    return numpy.exp(-integrate.quad(integrand_D, a,1)[0])

# a=1
# print(numpy.log(OmegaM(a)),integrate.quad(integrand, 1e-8, a)[0],numpy.log(OmegaM(a)) + integrate.quad(integrand, 1e-8, a)[0])
# a=1/(1+0.1)
# print(numpy.log(OmegaM(a)),integrate.quad(integrand, 1e-8, a)[0],numpy.log(OmegaM(a)) + integrate.quad(integrand, 1e-8, a)[0])
# a=0.5
# print(numpy.log(OmegaM(a)),integrate.quad(integrand, 1e-8, a)[0],numpy.log(OmegaM(a)) + integrate.quad(integrand, 1e-8, a)[0])
# a=1/(1+0.5)
# print(numpy.log(OmegaM(a)),integrate.quad(integrand, 1e-8, a)[0],numpy.log(OmegaM(a)) + integrate.quad(integrand, 1e-8, a)[0])
# a=0.5
# print(numpy.log(OmegaM(a)),integrate.quad(integrand, 1e-8, a)[0],numpy.log(OmegaM(a)) + integrate.quad(integrand, 1e-8, a)[0])
# wefwe

def Pvv(mu,f,D):
    return (f*D*mu*100)**2*matter[:,1]/matter[:,0]**2

def Pvv_l(mu,f,D):
    return 2*f*D*(mu*100)**2*matter[:,1]/matter[:,0]**2

# gg
b= 1.2
def Pgg(mu,f,D):
    return (b*D+f*D*mu**2)**2*matter[:,1]

def Pgg_l(mu,f,D):
    return 2*mu**2*(b*D+f*D*mu**2)*matter[:,1]

def Pgg_b(mu,f,D):
    return 2*(b*D+f*D*mu**2)*matter[:,1]

# gv
def Pgv(mu,f,D):
    return (b*D+f*D*mu**2)*(f*D*mu*100)*matter[:,1]/matter[:,0]

def Pgv_l(mu,f,D):
    return (mu**2*(f*D*mu*100)+(b*D+f*D*mu**2)*(mu*100))*matter[:,1]/matter[:,0]

def Pgv_b(mu,f,D):
    return (f*D*mu*100)*matter[:,1]/matter[:,0]

def finvp(f00,f11,f01,f00p,f11p,f01p):
    den = f00*f11 -f01**2
    return f11p /den - f11*(f00p*f11 + f00*f11p - 2*f01*f01p)/den**2

def Cmatrices(z,mu,ng,duration,sigm):
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

    pggs = Pgg(mu,f,D_)
    pgvs = Pgv(mu,f,D_)
    pvvs = Pvv(mu,f,D_)

    pggs_l = Pgg_l(mu,f,D_)
    pgvs_l = Pgv_l(mu,f,D_)
    pvvs_l = Pvv_l(mu,f,D_)

    pggs_b = Pgg_b(mu,f,D_)
    pgvs_b = Pgv_b(mu,f,D_)

    C=[]
    Cinv = []
    dCdl = []
    dCdb = []
    # Cinvs = []
    Cinvn = []
    Cinvsigmam = []
    for pgg, pgv, pvv, pgg_l, pgv_l, pvv_l, pgg_b, pgv_b  in zip(pggs,pgvs,pvvs,pggs_l,pgvs_l,pvvs_l,pggs_b,pgvs_b):
        C.append(numpy.array([[pgg+ninv,pgv],[pgv,pvv+nvinv]]))
        Cinv.append(numpy.linalg.inv(C[-1]))
        dCdl.append(numpy.array([[pgg_l,pgv_l],[pgv_l,pvv_l]])*OmegaM_a**0.55*lnfgfactor) # second term converts from fsigma8 to gamma
        dCdb.append(numpy.array([[pgg_b,pgv_b],[pgv_b,0]]))
        den = (pgg+ninv)*(pvv+nvinv)-pgv**2
        # Cinvs.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+nginv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+nginv]]))
        # Cinvs[-1]=-Cinvs[-1]*sigv**2/restrate/duration**2 #in terms of lnt not s = sigma^2/rate/s

        Cinvn.append(-1./den * numpy.array([[sigv2,0],[0,1]]) + (sigv2*(pgg+ninv)+(pvv+nvinv))/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))
        Cinvn[-1]=Cinvn[-1]/n #in terms of lnt  n**-2 * n = 1/n
        Cinvsigmam.append(1./den * numpy.array([[1,0],[0,0]]) + (pgg+ninv)/den**2 * numpy.array([[pvv+nvinv,-pgv],[-pgv,pgg+ninv]]))
        Cinvsigmam[-1] = Cinvsigmam[-1] * sigv_factor *3e5 * 2 * sigv / n
    return C, Cinv, dCdl,dCdb,Cinvn, Cinvsigmam

def traces(z,mu,ng,duration,sigm):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm)

    ll=[]
    lb=[]
    bb=[]

    lls=[]
    lbs=[]
    bbs=[]

    ll_ind=[]
    lb_ind=[]
    bb_ind=[]

    llsigM=[]
    lbsigM=[]
    bbsigM=[]   

    for C, Cinv, C_l,C_b,Cinvn,CinvsigM in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4],cmatrices[5]):
        ll.append(numpy.trace(Cinv@C_l@Cinv@C_l))
        bb.append(numpy.trace(Cinv@C_b@Cinv@C_b))
        lb.append(numpy.trace(Cinv@C_l@Cinv@C_b))

        lls.append(numpy.trace(Cinvn@ C_l@Cinv@C_l+Cinv@ C_l@Cinvn@C_l))
        bbs.append(numpy.trace(Cinvn@ C_b@Cinv@C_b+Cinv@ C_b@Cinvn@C_b))
        lbs.append(numpy.trace(Cinvn@ C_l@Cinv@C_b+Cinv@ C_l@Cinvn@C_b))

        llsigM.append(numpy.trace(CinvsigM@ C_l@Cinv@C_l+Cinv@ C_l@CinvsigM@C_l))
        bbsigM.append(numpy.trace(CinvsigM@ C_b@Cinv@C_b+Cinv@ C_b@CinvsigM@C_b))
        lbsigM.append(numpy.trace(CinvsigM@ C_l@Cinv@C_b+Cinv@ C_l@CinvsigM@C_b))

        ll_ind.append(C[0][0]**(-2)*C_l[0][0]**2+C[1][1]**(-2)*C_l[1][1]**2)
        bb_ind.append(C[0][0]**(-2)*C_b[0][0]**2+C[1][1]**(-2)*C_b[1][1]**2)
        lb_ind.append(C[0][0]**(-2)*C_l[0][0]*C_b[0][0]+C[1][1]**(-2)*C_l[1][1]*C_b[1][1])

    return numpy.array(ll),numpy.array(bb),numpy.array(lb),numpy.array(lls),numpy.array(bbs),numpy.array(lbs), \
        numpy.array(ll_ind),numpy.array(bb_ind),numpy.array(lb_ind),numpy.array(llsigM),numpy.array(bbsigM),numpy.array(lbsigM)

def muintegral(z,ng,duration,sigm):
    mus=numpy.arange(-1,1.001,0.1)

    ll=numpy.zeros((len(mus),len(matter[:,0])))
    bb=numpy.zeros((len(mus),len(matter[:,0])))
    bl=numpy.zeros((len(mus),len(matter[:,0])))
    lls=numpy.zeros((len(mus),len(matter[:,0])))
    bbs=numpy.zeros((len(mus),len(matter[:,0])))
    bls=numpy.zeros((len(mus),len(matter[:,0])))
    ll_ind=numpy.zeros((len(mus),len(matter[:,0])))
    bb_ind=numpy.zeros((len(mus),len(matter[:,0])))
    bl_ind=numpy.zeros((len(mus),len(matter[:,0])))
    llsigM=numpy.zeros((len(mus),len(matter[:,0])))
    bbsigM=numpy.zeros((len(mus),len(matter[:,0])))
    blsigM=numpy.zeros((len(mus),len(matter[:,0])))

    for i in range(len(mus)):
        dum = traces(z,mus[i],ng,duration,sigm)
        ll[i],bb[i],bl[i],lls[i],bbs[i],bls[i],ll_ind[i],bb_ind[i],bl_ind[i],llsigM[i],bbsigM[i],blsigM[i] = traces(z,mus[i],ng,duration,sigm)

    ll = numpy.trapz(ll,mus,axis=0)
    bb = numpy.trapz(bb,mus,axis=0)
    bl = numpy.trapz(bl,mus,axis=0)
    lls = numpy.trapz(lls,mus,axis=0)
    bbs = numpy.trapz(bbs,mus,axis=0)
    bls = numpy.trapz(bls,mus,axis=0)
    ll_ind = numpy.trapz(ll_ind,mus,axis=0)
    bb_ind = numpy.trapz(bb_ind,mus,axis=0)
    bl_ind = numpy.trapz(bl_ind,mus,axis=0)
    llsigM = numpy.trapz(llsigM,mus,axis=0)
    bbsigM = numpy.trapz(bbsigM,mus,axis=0)
    blsigM = numpy.trapz(blsigM,mus,axis=0)
    # ll,bb,bl, lls,bbs,bls, ll_ind,bb_ind,bl_ind = traces(z,mus[0],ng,duration,sigm)
    # ll = ll/2
    # bl = bl/2
    # bb = bb/2
    # lls =lls/2
    # bls = bls/2
    # bbs = bbs/2
    # ll_ind = ll_ind/2
    # bl_ind = bl_ind/2
    # bb_ind = bb_ind/2
    # for mu in mus[1:-1]:
    #     a,b,c,d,e,f,g,h,i = traces(z,mu,ng,duration,sigm)
    #     ll=ll+a
    #     bb=bb+b
    #     bl=bl+c
    #     lls=lls+d
    #     bbs=bbs+e
    #     bls=bls+f
    #     ll_ind=ll_ind+g
    #     bb_ind=bb_ind+h
    #     bl_ind=bl_ind+i
        
    # a,b,c,d,e,f,g,h,i = traces(z,mus[-1],ng,duration,sigm)
    # ll=ll+a/2
    # bb=bb+b/2
    # bl=bl+c/2
    # lls=lls+d/2
    # bbs=bbs+e/2
    # bls=bls+f/2
    # ll_ind=ll_ind+g/2
    # bb_ind=bb_ind+h/2
    # bl_ind=bl_ind+i/2
    
    return ll,bb,bl, lls,bbs,bls,ll_ind,bb_ind,bl_ind, llsigM,bbsigM,blsigM


def kintegral(z,zmax,ng,duration,sigm):
    kmin = 1./(zmax*3e3*2)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)

    ll,bb,bl, lls,bbs,bls,ll_ind,bb_ind,bl_ind, llsigM,bbsigM,blsigM = muintegral(z,ng,duration,sigm)
    ll = numpy.trapz(matter[:,0][w]**2*ll[w],matter[:,0][w])
    bb = numpy.trapz(matter[:,0][w]**2*bb[w],matter[:,0][w])
    bl = numpy.trapz(matter[:,0][w]**2*bl[w],matter[:,0][w])
    lls = numpy.trapz(matter[:,0][w]**2*lls[w],matter[:,0][w])
    bbs = numpy.trapz(matter[:,0][w]**2*bbs[w],matter[:,0][w])
    bls = numpy.trapz(matter[:,0][w]**2*bls[w],matter[:,0][w])
    ll_ind = numpy.trapz(matter[:,0][w]**2*ll_ind[w],matter[:,0][w])
    bb_ind = numpy.trapz(matter[:,0][w]**2*bb_ind[w],matter[:,0][w])
    bl_ind = numpy.trapz(matter[:,0][w]**2*bl_ind[w],matter[:,0][w])
    llsigM = numpy.trapz(matter[:,0][w]**2*llsigM[w],matter[:,0][w])
    bbsigM = numpy.trapz(matter[:,0][w]**2*bbsigM[w],matter[:,0][w])
    blsigM = numpy.trapz(matter[:,0][w]**2*blsigM[w],matter[:,0][w])

    return ll,bb,bl, lls,bbs,bls,ll_ind,bb_ind,bl_ind, llsigM,bbsigM,blsigM

# utility numbers
zmax_zint = 0.3
zs_zint = numpy.arange(0.01,0.3+0.00001,0.01) # in redshift space
rs_zint = cosmo.comoving_distance(zs_zint).value

def zintegral(zmax,ng,duration,sigm):
    # zs = numpy.arange(0.01,zmax+0.00001,0.01) # in redshift space


    # zs = zs*3e5/100
    # dz = 0.01*3e5/100

    w = zs_zint <= zmax
    zs = zs_zint[w]
    rs = rs_zint[w]

    ll=numpy.zeros(len(rs))
    bb=numpy.zeros(len(rs))
    bl=numpy.zeros(len(rs))
    lls=numpy.zeros(len(rs))
    bbs=numpy.zeros(len(rs))
    bls=numpy.zeros(len(rs))
    ll_ind=numpy.zeros(len(rs))
    bb_ind=numpy.zeros(len(rs))
    bl_ind=numpy.zeros(len(rs))
    llsigM=numpy.zeros(len(rs))
    bbsigM=numpy.zeros(len(rs))
    blsigM=numpy.zeros(len(rs))

    for i in range(len(rs)):
        ll[i],bb[i],bl[i],lls[i],bbs[i],bls[i],ll_ind[i],bb_ind[i],bl_ind[i],llsigM[i],bbsigM[i],blsigM[i] = kintegral(zs[i],zmax,ng,duration,sigm)

    ll = numpy.trapz(rs**2*ll,rs)
    bb = numpy.trapz(rs**2*bb,rs)
    bl = numpy.trapz(rs**2*bl,rs)
    lls = numpy.trapz(rs**2*lls,rs)
    bbs = numpy.trapz(rs**2*bbs,rs)
    bls = numpy.trapz(rs**2*bls,rs)
    ll_ind = numpy.trapz(rs**2*ll_ind,rs)
    bb_ind = numpy.trapz(rs**2*bb_ind,rs)
    bl_ind = numpy.trapz(rs**2*bl_ind,rs)
    llsigM = numpy.trapz(rs**2*llsigM,rs)
    bbsigM = numpy.trapz(rs**2*bbsigM,rs)
    blsigM = numpy.trapz(rs**2*blsigM,rs)

    # a,b,c,d,e,f,g,h,i = kintegral(rs[0],zmax,ng,duration,sigm)
    # deltars = rs[1]-rs[0]
    # ll=0.5*a*rs[0]**2*deltars
    # bb=0.5*b*rs[0]**2*deltars
    # bl=0.5*c*rs[0]**2*deltars
    # lls=0.5*d*rs[0]**2*deltars
    # bbs=0.5*e*rs[0]**2*deltars
    # bls=0.5*f*rs[0]**2*deltars
    # ll_ind=0.5*g*rs[0]**2*deltars
    # bb_ind=0.5*h*rs[0]**2*deltars
    # bl_ind=0.5*i*rs[0]**2*deltars

    # for r,rp in zip(rs[1:-1],rs[2:]):
    #     deltars = rp-r
    #     a,b,c,d,e,f,g,h,i = kintegral(r,zmax,ng,duration,sigm)        
    #     ll=ll+a*r**2*deltars
    #     bb=bb+b*r**2*deltars
    #     bl=bl+c*r**2*deltars
    #     lls=lls+d*r**2*deltars
    #     bbs=bbs+e*r**2*deltars
    #     bls=bls+ f*r**2*deltars
    #     ll_ind=ll_ind+g*r**2*deltars
    #     bb_ind=bb_ind+h*r**2*deltars
    #     bl_ind=bl_ind+i*r**2*deltars

    # a,b,c,d,e,f,g,h,i = kintegral(rs[-1],zmax,ng,duration,sigm)
    # deltars = rs[-1]-rs[0]
    # ll=ll+0.5*a*rs[-1]**2*deltars
    # bb=bb+0.5*b*rs[-1]**2*deltars
    # bl=bl+0.5*c*rs[-1]**2*deltars
    # lls=lls+0.5*d*rs[-1]**2*deltars
    # bbs=bbs+0.5*e*rs[-1]**2*deltars
    # bls=bls+ 0.5*f*rs[-1]**2*deltars
    # ll_ind=ll_ind+0.5*g*rs[-1]**2*deltars
    # bb_ind=bb_ind+0.5*h*rs[-1]**2*deltars
    # bl_ind=bl_ind+0.5*i*rs[-1]**2*deltars
    return ll, bb, bl, lls,bbs,bls,ll_ind, bb_ind, bl_ind, llsigM,bbsigM,blsigM


def set2():

    durations = [2,10]
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
            f00,f11,f10, _,_,_,_,_,_ = zintegral(zmax,ng,duration,sigm)
            var= numpy.linalg.inv(numpy.array([[f00,f10],[f10,f11]]))[0,0]*2*3.14/.75
            for z,r in zip(zs,rs):
                a=1./(1+z)
                drdz = 1/numpy.sqrt(OmegaM0/a**3 + (1-OmegaM0)) # 1/H
                _,_,_, f00s,f11s,f10s,_,_,_ = kintegral(z,zmax,ng,duration,sigm)
                dvardz.append(finvp(f00,f11,f10,f00s,f11s,f10s))
                dvardz[-1] = dvardz[-1]/r**2
                dvardz[-1] = dvardz[-1]*drdz  # z now has units of 100 km/s
                dvardz[-1] = dvardz[-1]* 3e3
            dvardz=numpy.array(dvardz)*2*3.14/.75
            plt.plot(zs,-dvardz/2/numpy.sqrt(var),label=r'{} $z_{{max}}={}$'.format(label,zmax),color=color,ls=ls)
    plt.xlabel('z')
    plt.yscale("log", nonposy='clip')
    plt.ylabel(r'$|\frac{d\sigma_{\gamma}}{dz}|$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dvardz.png')
    plt.clf()

# set2()


def set1():
    zmaxs = numpy.exp(numpy.arange(numpy.log(0.05),numpy.log(.300001),numpy.log(.3/.05)/8))
    durations = [2,10]
    labels = ['Two Years','Ten Years']
    var =[]
    dvards = []
    var_ind=[]
    dvardsigM = []
    for duration,label in zip(durations,labels):
        v_=[]
        vind_=[]
        dv_=[]
        dvsigM_=[]
        for zmax in zmaxs:
            f00,f11,f10, f00s,f11s,f10s,f00_ind,f11_ind,f10_ind, f00sigM,f11sigM,f10sigM = zintegral(zmax,ng,duration,sigm)
            dv_.append(finvp(f00,f11,f10,f00s,f11s,f10s))
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10],[f10,f11]]))[0,0])
            vind_.append(numpy.linalg.inv(numpy.array([[f00_ind,f10_ind],[f10_ind,f11_ind]]))[0,0])
            dvsigM_.append(finvp(f00,f11,f10,f00sigM,f11sigM,f10sigM))
        var.append(numpy.array(v_)*2*3.14/.75) #3/4
        dvards.append(numpy.array(dv_)*2*3.14/.75)
        var_ind.append(numpy.array(vind_)*2*3.14/.75) #3/4
        dvardsigM.append(numpy.array(dvsigM_)*2*3.14/.75)
 
    plt.plot(zmaxs,numpy.sqrt(var[0]),label='Two Years',color='red')
    plt.plot(zmaxs,numpy.sqrt(var[1]),label='Ten Years',color='black')
    plt.plot(zmaxs,numpy.sqrt(var_ind[0]),label='Two Years, Independent Surveys',color='red',ls=':')  
    plt.plot(zmaxs,numpy.sqrt(var_ind[1]),label='Ten Years, Independent Surveys',color='black',ls=':')   

    plt.xlabel(r'$z_{max}$')
    plt.ylim((0,0.07))
    plt.ylabel(r'$\sigma_{\gamma}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('var.png')

    plt.clf()
    plt.plot(zmaxs,-dvards[0]/2/numpy.sqrt(var[0]),label='Two Years',color='red')
    plt.plot(zmaxs,-dvards[1]/2/numpy.sqrt(var[1]),label='Ten Years',color='black')
    plt.legend()
    # plt.yscale("log", nonposy='clip')
    plt.ylim((0,plt.ylim()[1]))
    plt.xlabel(r'$z_{max}$')
    plt.ylabel(r'$|\frac{d\sigma_{\gamma}}{d\ln{t}}|$')
    plt.tight_layout()
    plt.savefig('dvardlnt.png')
    plt.clf()

    plt.plot(zmaxs,-dvardsigM[0]/2/numpy.sqrt(var[0]),label='Two Years',color='red')
    plt.plot(zmaxs,-dvardsigM[1]/2/numpy.sqrt(var[1]),label='Ten Years',color='black')
    plt.legend()
    # plt.yscale("log", nonposy='clip')
    plt.ylim((0,plt.ylim()[1]))
    plt.xlabel(r'$z_{max}$')
    plt.ylabel(r'$|\frac{d\sigma_{\gamma}}{d\sigma_M}|$')
    plt.tight_layout()
    plt.savefig('dvardsigM.png')
    plt.clf()

set1()