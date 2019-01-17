#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import matplotlib


matter = numpy.loadtxt("../../pv/dragan/matterpower.dat")

# vv
f=0.28**0.55

# SN rates
restrate = 0.65*  2.69e-5*(1/0.70)**3 # h^3/Mpc^3 yr^{-1}


sigm = 0.08

# galaxy density (TAIPAN Howlett et al.)
ng = 1e-3


def Pvv(mu):
    return (f*mu*100)**2*matter[:,1]/matter[:,0]**2

def Pvv_l(mu):
    return 2*f*(mu*100)**2*matter[:,1]/matter[:,0]**2

# gg
b= 1.2
def Pgg(mu):
    return (b+f*mu**2)**2*matter[:,1]

def Pgg_l(mu):
    return 2*mu**2*(b+f*mu**2)*matter[:,1]

def Pgg_b(mu):
    return 2*(b+f*mu**2)*matter[:,1]

# gv
def Pgv(mu):
    return (b+f*mu**2)*(f*mu*100)*matter[:,1]/matter[:,0]

def Pgv_l(mu):
    return (mu**2*(f*mu*100)+(b+f*mu**2)*(mu*100))*matter[:,1]/matter[:,0]

def Pgv_b(mu):
    return (f*mu*100)*matter[:,1]/matter[:,0]

def finvp(f00,f11,f01,f00p,f11p,f01p):
    den = f00*f11 -f01**2
    return f11p /den - f11*(f00p*f11 + f00*f11p - 2*f01*f01p)/den**2

def Cmatrices(z,mu,ng,duration,sigm):
    n = duration*restrate
    zz = z * 100./3e5
    sigv_factor = numpy.log(10)/5*zz/(1+zz)
    sigv =sigm * sigv_factor *3e5
    ninv = sigv**2/n

    dsdn = sigv**2/n**2

    nginv = 1./ng

    pggs = Pgg(mu)
    pgvs = Pgv(mu)
    pvvs = Pvv(mu)

    pggs_l = Pgg_l(mu)
    pgvs_l = Pgv_l(mu)
    pvvs_l = Pvv_l(mu)

    pggs_b = Pgg_b(mu)
    pgvs_b = Pgv_b(mu)

    C=[]
    Cinv = []
    dCdl = []
    dCdb = []
    Cinvs = []
    for pgg, pgv, pvv, pgg_l, pgv_l, pvv_l, pgg_b, pgv_b  in zip(pggs,pgvs,pvvs,pggs_l,pgvs_l,pvvs_l,pggs_b,pgvs_b):
        C.append(numpy.array([[pgg+nginv,pgv],[pgv,pvv+ninv]]))
        Cinv.append(numpy.linalg.inv(C[-1]))
        dCdl.append(numpy.array([[pgg_l,pgv_l],[pgv_l,pvv_l]]))
        dCdb.append(numpy.array([[pgg_b,pgv_b],[pgv_b,0]]))
        den = (pgg+nginv)*(pvv+ninv)-pgv**2
        Cinvs.append(1./den * numpy.array([[1,0],[0,0]]) - (pgg+nginv)/den**2 * numpy.array([[pvv+ninv,-pgv],[-pgv,pgg+nginv]]))
        Cinvs[-1]=Cinvs[-1]*sigv**2/n**2*restrate*duration
    return C, Cinv, dCdl,dCdb,Cinvs

def traces(z,mu,ng,duration,sigm):

    cmatrices = Cmatrices(z,mu,ng,duration,sigm)

    ll=[]
    lb=[]
    bb=[]

    lls=[]
    lbs=[]
    bbs=[]

    for C, Cinv, C_l,C_b,Cinvs in zip(cmatrices[0],cmatrices[1],cmatrices[2],cmatrices[3],cmatrices[4]):
        ll.append(numpy.trace(Cinv@C_l@Cinv@C_l))
        bb.append(numpy.trace(Cinv@C_b@Cinv@C_b))
        lb.append(numpy.trace(Cinv@C_l@Cinv@C_b))

        lls.append(numpy.trace(Cinvs@ C_l@Cinv@C_l+Cinv@ C_l@Cinvs@C_l))
        bbs.append(numpy.trace(Cinvs@ C_b@Cinv@C_b+Cinv@ C_b@Cinvs@C_b))
        lbs.append(numpy.trace(Cinvs@ C_l@Cinv@C_b+Cinv@ C_l@Cinvs@C_b))
    return numpy.array(ll),numpy.array(bb),numpy.array(lb),numpy.array(lls),numpy.array(bbs),numpy.array(lbs)

def muintegral(z,ng,duration,sigm):
    mus=numpy.arange(-1,1.001,0.1)
    ll,bb,bl, lls,bbs,bls = traces(z,mus[0],ng,duration,sigm)
    ll = ll/2
    bl = bl/2
    bb = bb/2
    lls =lls/2
    bls = bls/2
    bbs = bbs/2
    for mu in mus[1:-1]:
        a,b,c,d,e,f = traces(z,mu,ng,duration,sigm)
        ll=ll+a
        bb=bb+b
        bl=bl+c
        lls=lls+d
        bbs=bbs+e
        bls=bls+f
        
    a,b,c,d,e,f = traces(z,mus[-1],ng,duration,sigm)
    ll=ll+a/2
    bb=bb+b/2
    bl=bl+c/2
    lls=lls+d/2
    bbs=bbs+e/2
    bls=bls+f/2
    
    return 0.1*ll,0.1*bb,0.1*bl, 0.1*lls,0.1*bbs,0.1*bls


def kintegral(z,zmax,ng,duration,sigm):
    kmin = 1./(zmax*3e3*2)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)

    ll,bb,bl, lls,bbs,bls = muintegral(z,ng,duration,sigm)
    ll = numpy.trapz(matter[:,0][w]**2*ll[w],matter[:,0][w])
    bb = numpy.trapz(matter[:,0][w]**2*bb[w],matter[:,0][w])
    bl = numpy.trapz(matter[:,0][w]**2*bl[w],matter[:,0][w])
    lls = numpy.trapz(matter[:,0][w]**2*lls[w],matter[:,0][w])
    bbs = numpy.trapz(matter[:,0][w]**2*bbs[w],matter[:,0][w])
    bls = numpy.trapz(matter[:,0][w]**2*bls[w],matter[:,0][w])

    return ll,bb,bl, lls,bbs,bls

def zintegral(zmax,ng,duration,sigm):
    zs = numpy.arange(0.01,zmax+0.00001,0.01)
    zs = zs*3e5/100
    dz = 0.01*3e5/100
    a,b,c,d,e,f = kintegral(zs[0],zmax,ng,duration,sigm)
    ll=0.5*a*zs[0]**2
    bb=0.5*b*zs[0]**2
    bl=0.5*c*zs[0]**2
    lls=0.5*d*zs[0]**2
    bbs=0.5*e*zs[0]**2
    bls=0.5*f*zs[0]**2

    for z in zs[1:-1]:
        a,b,c,d,e,f = kintegral(z,zmax,ng,duration,sigm)        
        ll=ll+a*z**2
        bb=bb+b*z**2
        bl=bl+c*z**2
        lls=lls+d*z**2
        bbs=bbs+e*z**2
        bls=bls+ f*z**2

    a,b,c,d,e,f = kintegral(zs[-1],zmax,ng,duration,sigm)
    ll=ll+0.5*a*zs[-1]**2
    bb=bb+0.5*b*zs[-1]**2
    bl=bl+0.5*c*zs[-1]**2
    lls=lls+0.5*d*zs[-1]**2
    bbs=bbs+0.5*e*zs[-1]**2
    bls=bls+ 0.5*f*zs[-1]**2

    return dz*ll, dz*bb, dz*bl, dz*lls,dz*bbs,dz*bls


def set2():
    durations = [2,10]
    labels = ['Two Years','Ten Years']
    colors = ['red','black']

    zmaxs=[0.05,0.2]
    lss = [':','-']
    for ls,zmax in zip(lss, zmaxs):
        zs = numpy.arange(0.01,zmax+0.00001,0.02)
        zs_d = zs*3e5/100
        for duration,label,color in zip(durations,labels,colors):
            dvardz=[]
            f00,f11,f10, _,_,_ = zintegral(zmax,ng,duration,sigm)
            var= numpy.linalg.inv(numpy.array([[f00,f10],[f10,f11]]))[0,0]*2*3.14/.75
            for z in zs_d:
                _,_,_, f00s,f11s,f10s = kintegral(z,zmax,ng,duration,sigm)
                dvardz.append(finvp(f00,f11,f10,f00s,f11s,f10s))
                dvardz[-1] = dvardz[-1]/z**2
            dvardz=numpy.array(dvardz)*2*3.14/.75
            plt.plot(zs,zmax*dvardz/2/numpy.sqrt(var),label=r'{} $z_{{max}}={}$'.format(label,zmax),color=color,ls=ls)
    plt.xlabel('z')
    plt.yscale("log", nonposy='clip')
    plt.ylabel(r'$\frac{d\sigma_{f\sigma_8}}{dz} z_{max} (f \sigma_8)^{-1}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dvardz.png')

set2()

def set1():
    zmaxs = numpy.exp(numpy.arange(numpy.log(0.05),numpy.log(.300001),numpy.log(.3/.05)/8))
    durations = [2,10]
    labels = ['Two Years','Ten Years']
    var =[]
    dvards = []
    for duration,label in zip(durations,labels):
        v_=[]
        dv_=[]
        for zmax in zmaxs:
            f00,f11,f10, f00s,f11s,f10s = zintegral(zmax,ng,duration,sigm)
            dv_.append(finvp(f00,f11,f10,f00s,f11s,f10s))
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10],[f10,f11]]))[0,0])
        var.append(numpy.array(v_)*2*3.14/.75) #3/4
        dvards.append(numpy.array(dv_)*2*3.14/.75)

    plt.plot(zmaxs,numpy.sqrt(var[0])/f,label='Two Years',color='red')
    plt.plot(zmaxs,numpy.sqrt(var[1])/f,label='Ten Years',color='black')
    plt.xlabel(r'$z_{max}$')
    plt.ylim((0,0.1))
    plt.ylabel(r'$\sigma_{f\sigma_8}(f \sigma_8)^{-1}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('var.png')

    plt.clf()
    plt.plot(zmaxs,dvards[0]/2/numpy.sqrt(var[0])/f,label='Two Years',color='red')
    plt.plot(zmaxs,dvards[1]/2/numpy.sqrt(var[1])/f,label='Ten Years',color='black')
    plt.legend()
    plt.xlabel(r'$z_{max}$')
    plt.ylabel(r'$\frac{d\sigma_{f\sigma_8}}{d\ln{t}}(f\sigma_8)^{-1}$')
    plt.tight_layout()
    plt.savefig('dvards.png')

set1()