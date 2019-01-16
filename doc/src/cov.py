#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
import matplotlib


matter = numpy.loadtxt("../../pv/dragan/matterpower.dat")

# vv
f=0.28**0.55

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


# SN rates
restrate =  2.69e-5*(1/0.70)**3 # h^3/Mpc^3 yr^{-1}
twoyeardensity = 0.65*2*restrate
tenyeardensity = 0.65*10*restrate

sigm = 0.08

# galaxy density (TAIPAN Howlett et al.)
ng = 1e-3

mus=[1,0.5]
#lss=[':','-']

# sigv_factor = numpy.log(10)/5*zs/(1+zs)
# sigvs =sigm * sigv_factor *3e5

# the four traces in an array.  trace is a function of (z)
def trace(z, mu, sigm, n,ng,independent=False):

    pvv = Pvv(mu)
    pgg = Pgg(mu)
    pvg = Pgv(mu)

    pvv_l = Pvv_l(mu)
    pgg_l = Pgg_l(mu)
    pvg_l = Pgv_l(mu)  

    pgg_b = Pgg_b(mu)
    pvg_b = Pgv_b(mu)

    sigv_factor = numpy.log(10)/5*z/(1+z)
    sigv =sigm * sigv_factor *3e5
    shot = sigv**2/n
    trace_ll = []
    trace_lb = []
    trace_bl = []
    trace_bb = []
    for  k, p00, p01, p11,p00_l, p01_l, p11_l,p00_b, p01_b in zip(matter[:,0],pgg,pvg,pvv,pgg_l,pvg_l,pvv_l,pgg_b,pvg_b):
        if independent:
            C = numpy.array([[p00+1/ng,0],[0,p11+shot]])
            Cinv = numpy.linalg.inv(C)
            C_l = numpy.array([[p00_l,0],[0,p11_l]])
            C_b = numpy.array([[p00_b,0],[0,0]])
        else:
            C = numpy.array([[p00+1/ng,p01],[p01,p11+shot]])
            Cinv = numpy.linalg.inv(C)
            C_l = numpy.array([[p00_l,p01_l],[p01_l,p11_l]])
            C_b = numpy.array([[p00_b,p01_b],[p01_b,0]])            
        trace_ll.append(numpy.trace(Cinv@ C_l@Cinv@C_l))
        trace_lb.append(numpy.trace(Cinv@ C_l@Cinv@C_b))
        trace_bl.append(numpy.trace(Cinv@ C_b@Cinv@C_l))
        trace_bb.append(numpy.trace(Cinv@ C_b@Cinv@C_b))
    return numpy.array(trace_ll),numpy.array(trace_lb),numpy.array(trace_bl),numpy.array(trace_bb)

def muintegral(z,sigm, n,ng,independent=False):
    mus=numpy.arange(-1,1.001,0.1)
    trac = trace(z, mus[0], sigm, n, ng,independent=independent)
    ans0 = 0.5*trac[0]
    ans1 = 0.5*trac[1]
    ans2 = 0.5*trac[2]
    ans3 = 0.5*trac[3]
    for mu in mus[1:-1]:
        trac= trace(z, mu, sigm, n, ng,independent=independent)
        ans0 = ans0+trac[0]
        ans1 = ans1+trac[1]
        ans2 = ans2+trac[2]
        ans3 = ans3+trac[3]
    trac = trace(z, mus[-1], sigm, n, ng,independent=independent)
    ans0 = ans0+0.5*trac[0]
    ans1 = ans1+0.5*trac[1]
    ans2 = ans2+0.5*trac[2]
    ans3 = ans3+0.5*trac[3]

    return 0.1*ans0,0.1*ans1,0.1*ans2,0.1*ans3


def kintegral(zmax,z, sigm, n,ng,independent=False):
    # 00 term
    kmin = 1./(zmax*3e3*2)
    kmax = 0.1
    w = numpy.logical_and(matter[:,0] >= kmin, matter[:,0]< kmax)
    trac = muintegral(z,sigm, n,ng,independent=independent)
    return numpy.trapz(matter[:,0][w]**2*trac[0][w],matter[:,0][w]), \
        numpy.trapz(matter[:,0][w]**2*trac[1][w],matter[:,0][w]), \
        numpy.trapz(matter[:,0][w]**2*trac[2][w],matter[:,0][w]),\
        numpy.trapz(matter[:,0][w]**2*trac[3][w],matter[:,0][w])

def zintegral(zmax,sigm, n,ng,independent=False):
    zs = numpy.arange(0.01,zmax+0.00001,0.01)
    arr0 = []
    arr1 = []
    arr2 = []
    arr3 = []
    for z in zs:
        ki =kintegral(zmax,z, sigm, n, ng,independent=independent)
        arr0.append(ki[0])
        arr1.append(ki[1])
        arr2.append(ki[2])
        arr3.append(ki[3]) 
    arr0 = numpy.array(arr0)
    arr1 = numpy.array(arr1)
    arr2 = numpy.array(arr2)
    arr3 = numpy.array(arr3)
    return numpy.trapz(zs**2*arr0,zs),numpy.trapz(zs**2*arr1,zs),numpy.trapz(zs**2*arr2,zs),numpy.trapz(zs**2*arr3,zs)


zmaxs = numpy.arange(0.05,0.301,0.025)
# zmaxs = numpy.log(numpy.arange(1,10.01,2))*0.25/numpy.log(10)+0.05
noiseless = []
twoyear = []
tenyear = []
twoyearind = []
tenyearind = []
independent = []
for zmax in zmaxs:
    zi = zintegral(zmax,0, tenyeardensity,ng)
    F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    noiseless.append(numpy.linalg.inv(F)[0,0])
    # zi = zintegral(zmax,0, tenyeardensity, pgg,pvg,pvv,pgg_l,pvg_l,pvv_l,pgg_b,pvg_b,ng,independent=True)
    # F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    # independent.append(numpy.linalg.inv(F)[0,0])        
    zi = zintegral(zmax,sigm, twoyeardensity,ng)
    F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    twoyear.append(numpy.linalg.inv(F)[0,0])
    zi = zintegral(zmax,sigm, tenyeardensity,ng)
    F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    tenyear.append(numpy.linalg.inv(F)[0,0])
    zi = zintegral(zmax,sigm, twoyeardensity,ng,independent=True)
    F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    twoyearind.append(numpy.linalg.inv(F)[0,0])
    zi = zintegral(zmax,sigm, tenyeardensity,ng,independent=True)
    F = numpy.array([[zi[0],zi[1]],[zi[2],zi[3]]])
    tenyearind.append(numpy.linalg.inv(F)[0,0])

noiseless=numpy.array(noiseless)
twoyear=numpy.array(twoyear)
tenyear= numpy.array(tenyear)
twpyearind= numpy.array(twoyearind)
tenyearind= numpy.array(tenyearind)
independent = numpy.array(independent)
plt.plot(zmaxs,numpy.sqrt(twoyear/noiseless),label='Two Year',color='red')
plt.plot(zmaxs,numpy.sqrt(tenyear/noiseless),label='Ten Year',color='blue')
# plt.plot(zmaxs,numpy.sqrt(independent/noiseless),label='Independent',color='black',ls=':')
plt.plot(zmaxs,numpy.sqrt(twoyearind/noiseless),label='Two Year Independent',color='red',ls=':')  
plt.plot(zmaxs,numpy.sqrt(tenyearind/noiseless),label='Ten Year Independent',color='blue',ls=':')    
plt.ylabel(r'$\sigma_{f\sigma_8}/\sigma_{f\sigma_8,min}$')
plt.ylim((1,7))
plt.yscale("log", nonposx='clip')
plt.legend()
plt.xlabel(r'$z_{max}$')
plt.tight_layout()
plt.savefig('new.png')
#     plt.show()

#     wefe

#     shotless=[]
#     nocov=[]
  
#     for p00, p01, p11,p00_l, p01_l, p11_l in zip(pgg,pgv,pvv,pgg_l,pgv_l,pvv_l):

#         C_l = numpy.array([[p00_l,p01_l],[p01_l,p11_l]])
#         C = numpy.array([[p00+1/galdensity,p01],[p01,p11]])
#         Cinv = numpy.linalg.inv(C)
#         shotless.append(numpy.trace(Cinv@ C_l@Cinv@C_l))

#         C_l = numpy.array([[p00_l,0],[0,p11_l]])
#         C = numpy.array([[p00+1/galdensity,0],[0,p11]])
#         Cinv = numpy.linalg.inv(C)
#         nocov.append(numpy.trace(Cinv@ C_l@Cinv@C_l))

#     shotless = numpy.array(shotless)
#     nocov=numpy.array(nocov)
#     ax.plot(matter[:,0],nocov/shotless,label=r'No Covariance',color='black',ls='--')
#     for sigv,z,color in zip(sigvs,zs,colors):
#         twoyear=[]
#         tenyear=[]
#         for p00, p01, p11,p00_l, p01_l, p11_l in zip(pgg,pgv,pvv,pgg_l,pgv_l,pvv_l):

#             C_l = numpy.array([[p00_l,p01_l],[p01_l,p11_l]])
#             C = numpy.array([[p00+1/galdensity,p01],[p01,p11+sigv**2/twoyeardensity]])
#             Cinv = numpy.linalg.inv(C)
#             twoyear.append(numpy.trace(Cinv@ C_l@Cinv@C_l))
#             C = numpy.array([[p00+1/galdensity,p01],[p01,p11+sigv**2/tenyeardensity]])
#             Cinv = numpy.linalg.inv(C)
#             tenyear.append(numpy.trace(Cinv@ C_l@Cinv@C_l))
#         twoyear = numpy.array(twoyear)
#         tenyear = numpy.array(tenyear)
#         ax.plot(matter[:,0],twoyear/shotless,label=r'$\sigma(z={})$, Two Year'.format(z),color=color,ls=':')
#         ax.plot(matter[:,0],tenyear/shotless,label=r'$\sigma(z={})$, Ten Year'.format(z),linewidth=2,color=color)
#     dum=-6.7
#     ax.plot([1./(zs[0]*3e3*2),0.1],[numpy.exp(dum*.94),numpy.exp(dum*.94)],color=colors[0])
#     ax.plot([1./(zs[1]*3e3*2),0.1],[numpy.exp(dum*1.02),numpy.exp(dum*1.02)],color=colors[1])
#     ax.plot([1./(zs[2]*3e3*2),0.1],[numpy.exp(dum*1.1),numpy.exp(dum*1.1)],color=colors[2])
#     ax.text(0.1, numpy.exp(dum*.94),r'$z_{{max}}={}$'.format(zs[0]),color=colors[0])
#     ax.text(0.1, numpy.exp(dum*1.02),r'$z_{{max}}={}$'.format(zs[1]),color=colors[1])
#     ax.text(0.1, numpy.exp(dum*1.1),r'$z_{{max}}={}$'.format(zs[2]),color=colors[2])
#     # pl.format(z)t.plot(matter[:,0],shotless,label='Sample Variance Limit',color='black',linewidth=2)

# # shotless = numpy.array(shotless)
# # twoyear = numpy.array(twoyear)
# # tenyear = numpy.array(tenyear)

# # plt.plot(matter[:,0],shotless,label='Sample Variance Limit')
# # plt.plot(matter[:,0],twoyear,label='Two Year')
# # plt.plot(matter[:,0],tenyear,label='Ten Year')
#     ax.set_xlim((6e-4,.2))
#     ax.set_ylim((.5e-3,1))
#     ax.set_yscale("log", nonposx='clip')
#     ax.set_xscale("log", nonposx='clip')
#     ax.text(0.2,0.4,r'$\mu={}$'.format(mu),transform=ax.transAxes,fontsize=14)
#     # ax.set_ylabel(r'Tr$\left(C^{-1}C_{,f\sigma_8}C^{-1}C_{,f\sigma_8}\right)$')
#     ax.set_ylabel(r'$I/I_{max}$')
# # plt.legend()
# axs[1].legend(prop={'size': 6})
# axs[2].set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')

plt.clf()
fig, axs = plt.subplots(1,1,sharex=True,figsize=(8,5))

colors=['blue','green','red']
zs = numpy.array([0.05,0.1,.2])
axs.plot(matter[:,0],matter[:,0]**3*Pvv(1),label=r'$\mu=1$')
axs.plot(matter[:,0],matter[:,0]**3*Pvv(0.5),label=r'$\mu=0.5$')
axs.set_ylabel(r'$k^3P_{vv}$[km$^2$s$^{-2}$]')
# axs.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
dum=1
axs.plot([1./(zs[0]*3e3*2),0.1],[numpy.exp(dum*10.6),numpy.exp(dum*10.6)],color=colors[0])
axs.plot([1./(zs[1]*3e3*2),0.1],[numpy.exp(dum*10.3),numpy.exp(dum*10.3)],color=colors[1])
axs.plot([1./(zs[2]*3e3*2),0.1],[numpy.exp(dum*10),numpy.exp(dum*10)],color=colors[2])
axs.text(0.1, numpy.exp(dum*10.6),r'$z_{{max}}={}$'.format(zs[0]),color=colors[0])
axs.text(0.1, numpy.exp(dum*10.3),r'$z_{{max}}={}$'.format(zs[1]),color=colors[1])
axs.text(0.1, numpy.exp(dum*10),r'$z_{{max}}={}$'.format(zs[2]),color=colors[2])
axs.set_xlim((6e-4,.2))
# axs.set_ylim((1e6,1e8))
axs.set_ylim((.2e5,.2e7))
axs.set_yscale("log", nonposx='clip')
axs.set_xscale("log", nonposx='clip')
axs.legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('new2.png')
wefwe

zs = numpy.array([0.05,0.1,.2])
fig, axs = plt.subplots(3,1,sharex=True,figsize=(8,5))

for mu,ax in zip(mus,axs[1:]):
    shotless=[]
    nocov=[]
    pvv = Pvv(mu)
    pgg = Pgg(mu)
    pgv = Pgv(mu)

    pvv_l = Pvv_l(mu)
    pgg_l = Pgg_l(mu)
    pgv_l = Pgv_l(mu)    
    for p00, p01, p11,p00_l, p01_l, p11_l in zip(pgg,pgv,pvv,pgg_l,pgv_l,pvv_l):

        C_l = numpy.array([[p00_l,p01_l],[p01_l,p11_l]])
        C = numpy.array([[p00+1/galdensity,p01],[p01,p11]])
        Cinv = numpy.linalg.inv(C)
        shotless.append(numpy.trace(Cinv@ C_l@Cinv@C_l))

        C_l = numpy.array([[p00_l,0],[0,p11_l]])
        C = numpy.array([[p00+1/galdensity,0],[0,p11]])
        Cinv = numpy.linalg.inv(C)
        nocov.append(numpy.trace(Cinv@ C_l@Cinv@C_l))

    shotless = numpy.array(shotless)
    nocov=numpy.array(nocov)
    ax.plot(matter[:,0],nocov/shotless,label=r'No Covariance',color='black',ls='--')
    for sigv,z,color in zip(sigvs,zs,colors):
        twoyear=[]
        tenyear=[]
        for p00, p01, p11,p00_l, p01_l, p11_l in zip(pgg,pgv,pvv,pgg_l,pgv_l,pvv_l):

            C_l = numpy.array([[p00_l,p01_l],[p01_l,p11_l]])
            C = numpy.array([[p00+1/galdensity,p01],[p01,p11+sigv**2/twoyeardensity]])
            Cinv = numpy.linalg.inv(C)
            twoyear.append(numpy.trace(Cinv@ C_l@Cinv@C_l))
            C = numpy.array([[p00+1/galdensity,p01],[p01,p11+sigv**2/tenyeardensity]])
            Cinv = numpy.linalg.inv(C)
            tenyear.append(numpy.trace(Cinv@ C_l@Cinv@C_l))
        twoyear = numpy.array(twoyear)
        tenyear = numpy.array(tenyear)
        ax.plot(matter[:,0],twoyear/shotless,label=r'$\sigma(z={})$, Two Year'.format(z),color=color,ls=':')
        ax.plot(matter[:,0],tenyear/shotless,label=r'$\sigma(z={})$, Ten Year'.format(z),linewidth=2,color=color)
    dum=-6.7
    ax.plot([1./(zs[0]*3e3*2),0.1],[numpy.exp(dum*.94),numpy.exp(dum*.94)],color=colors[0])
    ax.plot([1./(zs[1]*3e3*2),0.1],[numpy.exp(dum*1.02),numpy.exp(dum*1.02)],color=colors[1])
    ax.plot([1./(zs[2]*3e3*2),0.1],[numpy.exp(dum*1.1),numpy.exp(dum*1.1)],color=colors[2])
    ax.text(0.1, numpy.exp(dum*.94),r'$z_{{max}}={}$'.format(zs[0]),color=colors[0])
    ax.text(0.1, numpy.exp(dum*1.02),r'$z_{{max}}={}$'.format(zs[1]),color=colors[1])
    ax.text(0.1, numpy.exp(dum*1.1),r'$z_{{max}}={}$'.format(zs[2]),color=colors[2])
    # pl.format(z)t.plot(matter[:,0],shotless,label='Sample Variance Limit',color='black',linewidth=2)

# shotless = numpy.array(shotless)
# twoyear = numpy.array(twoyear)
# tenyear = numpy.array(tenyear)

# plt.plot(matter[:,0],shotless,label='Sample Variance Limit')
# plt.plot(matter[:,0],twoyear,label='Two Year')
# plt.plot(matter[:,0],tenyear,label='Ten Year')
    ax.set_xlim((6e-4,.2))
    ax.set_ylim((.5e-3,1))
    ax.set_yscale("log", nonposx='clip')
    ax.set_xscale("log", nonposx='clip')
    ax.text(0.2,0.4,r'$\mu={}$'.format(mu),transform=ax.transAxes,fontsize=14)
    # ax.set_ylabel(r'Tr$\left(C^{-1}C_{,f\sigma_8}C^{-1}C_{,f\sigma_8}\right)$')
    ax.set_ylabel(r'$I/I_{max}$')
# plt.legend()
axs[1].legend(prop={'size': 6})
axs[2].set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
axs[0].plot(matter[:,0],matter[:,0]**3*Pvv(1),label=r'$\mu=1$')
axs[0].plot(matter[:,0],matter[:,0]**3*Pvv(0.5),label=r'$\mu=0.5$')
axs[0].set_ylabel(r'$k^3P_{vv}$[km$^2$s$^{-2}$]')
# axs[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
dum=1
axs[0].plot([1./(zs[0]*3e3*2),0.1],[numpy.exp(dum*10.6),numpy.exp(dum*10.6)],color=colors[0])
axs[0].plot([1./(zs[1]*3e3*2),0.1],[numpy.exp(dum*10.3),numpy.exp(dum*10.3)],color=colors[1])
axs[0].plot([1./(zs[2]*3e3*2),0.1],[numpy.exp(dum*10),numpy.exp(dum*10)],color=colors[2])
axs[0].text(0.1, numpy.exp(dum*10.6),r'$z_{{max}}={}$'.format(zs[0]),color=colors[0])
axs[0].text(0.1, numpy.exp(dum*10.3),r'$z_{{max}}={}$'.format(zs[1]),color=colors[1])
axs[0].text(0.1, numpy.exp(dum*10),r'$z_{{max}}={}$'.format(zs[2]),color=colors[2])
axs[0].set_xlim((6e-4,.2))
# axs[0].set_ylim((1e6,1e8))
axs[0].set_ylim((.2e5,.2e7))
axs[0].set_yscale("log", nonposx='clip')
axs[0].set_xscale("log", nonposx='clip')
axs[0].legend()
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('power.png')
wefwe
plt.clf()


fig, axs = plt.subplots(2,1,sharex=True)

for mu,ax in zip(mus,axs):
    shotless=[]
    nocov=[]
    pvv = Pvv(mu)
    # for p00, p01, p11 in zip(pgg,pgv,pvv):
    #     shotless.append((p00+1/galdensity)*p11 - p01**2)
    #     nocov.append(((p00+1/galdensity)*p11))
    # shotless = numpy.array(shotless)
    # nocov=numpy.array(nocov)
    # ax.plot(matter[:,0],nocov/shotless,label=r'No Covariance',color='black',ls='--')
    #for sigv,z,color in zip(sigvs,zs,colors):
    ans = matter[:,0]**3*pvv
    ax.plot(matter[:,0],ans)
    # for sigv,z,color in zip(sigvs,zs,colors):
    #     twoyear = pvv + sigv**2/twoyeardensity
    #     tenyear = pvv
    #     # twoyear=[]
    #     # tenyear=[]
    #     # for p11 in pvv:
    #     #     twoyear.append((p11+sigv**2/twoyeardensity) - p01**2)
    #     #     tenyear.append((p00+1/galdensity)*(p11+sigv**2/tenyeardensity) - p01**2)
    #     # twoyear = numpy.array(twoyear)
    #     # tenyear = numpy.array(tenyear)
    #     ax.plot(matter[:,0],twoyear,label=r'$\sigma(z={})$, Two Year'.format(z),color=color,ls=':')
    #     ax.plot(matter[:,0],tenyear,label=r'$\sigma(z={})$, Ten Year'.format(z),linewidth=2,color=color)
    # dum=.3
    # ax.plot([1./(zs[0]*3e3*2),0.1],[numpy.exp(dum*3),numpy.exp(dum*3)],color=colors[0])
    # ax.plot([1./(zs[1]*3e3*2),0.1],[numpy.exp(dum*2),numpy.exp(dum*2)],color=colors[1])
    # ax.plot([1./(zs[2]*3e3*2),0.1],[numpy.exp(dum),numpy.exp(dum)],color=colors[2])
    # ax.text(0.1, numpy.exp(dum*3),r'$z_{{max}}={}$'.format(zs[0]),color=colors[0])
    # ax.text(0.1, numpy.exp(dum*2),r'$z_{{max}}={}$'.format(zs[1]),color=colors[1])
    # ax.text(0.1, numpy.exp(dum),r'$z_{{max}}={}$'.format(zs[2]),color=colors[2])
    # pl.format(z)t.plot(matter[:,0],shotless,label='Sample Variance Limit',color='black',linewidth=2)

# shotless = numpy.array(shotless)
# twoyear = numpy.array(twoyear)
# tenyear = numpy.array(tenyear)

# plt.plot(matter[:,0],shotless,label='Sample Variance Limit')
# plt.plot(matter[:,0],twoyear,label='Two Year')
# plt.plot(matter[:,0],tenyear,label='Ten Year')
    ax.set_xlim((5e-4,.5))
    # ax.set_ylim((1,100))
    ax.set_yscale("log", nonposx='clip')
    ax.set_xscale("log", nonposx='clip')
    ax.set_title(r'$\mu={}$'.format(mu))
    ax.set_ylabel(r'$P$')
# plt.legend()
axs[0].legend(prop={'size': 6})
axs[1].set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
plt.tight_layout()
plt.show()


# plt.clf()
# plt.plot(matter[:,0],shotless,label='Sample Variance Limit')
# plt.xlim((5e-4,.5))
# plt.xscale("log", nonposx='clip')
# plt.show()



font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

fig, ax2 = plt.subplots(1, 1)


ax2.plot(matter[:,0],(f*100)**2*matter[:,1]/matter[:,0]**2,label=r'$P_{vv}(k,\mu=1,z=0)$',color='black')
ax2.plot(matter[:,0],(f*0.5*100)**2*matter[:,1]/matter[:,0]**2,label=r'$P_{vv}(k,\mu=0.5,z=0)$',color='black',ls=':')
ax2.plot(matter[:,0],(f*0.1*100)**2*matter[:,1]/matter[:,0]**2,label=r'$P_{vv}(k,\mu=0.1,z=0)$',color='black',ls='--')
ax2.text(1e-2,.8e12,r'$P_{vv}(\mu=1)$')
ax2.text(.55e-3,5e12,r'$P_{vv}(\mu=0.5)$')
ax2.text(1.05e-3,1.05e11,r'$P_{vv}(\mu=0.1)$')


# ax2.plot([1./(zs[2]*3e3*2),.1],[1e13,1e13],color='red')
# ax2.text(0.11, 1e13,r'$k_{min}(z=0.2)$',color='red')
# ax2.plot([1./(zs[1]*3e3*2),.1],[5e12,5e12],color='green')
# ax2.text(0.11, 5e12,r'$k_{min}(z=0.1)$',color='green')
# ax2.plot([1./(zs[0]*3e3*2),.1],[1e12,1e12],color='blue')
# ax2.text(0.11, 1e12,r'$k_{min}(z=0.05)$',color='blue')
# ax2.text(0.1, 1.1e11,r'$k_{max}$',color='grey')

#ax2.annotate(r'$k_{min}(z=0.05)$', xy=(1./(zs[0]*3e3)+1, 1e12), xytext=(1./(zs[0]*3e3), 1e12),arrowprops=dict(facecolor='blue', shrink=0.05),)
# ax2.axvline(x=1./(zs[0]*3e3)/2,color='blue',ls=':')
# ax2.axvline(x=1./(zs[1]*3e3)/2,color='green',ls=':')
# ax2.axvline(x=1./(zs[2]*3e3)/2,color='red',ls=':')
# ax2.axvline(x=.1,color='gray',ls=':')

ax2.plot([1./(zs[0]*3e3*2),0.1],[sigv[0]**2/twoyeardensity,sigv[0]**2/twoyeardensity],color='blue',ls='--')
ax2.plot([1./(zs[1]*3e3*2),0.1],[sigv[1]**2/twoyeardensity,sigv[1]**2/twoyeardensity],color='green',ls='--')
ax2.plot([1./(zs[2]*3e3*2),0.1],[sigv[2]**2/twoyeardensity,sigv[2]**2/twoyeardensity],color='red',ls='--')
ax2.text(0.08,sigv[0]**2/twoyeardensity+1e9,r'$\sigma^2/n(z=0.05)$, 2 yr',color='blue')
ax2.text(0.08,sigv[1]**2/twoyeardensity+2e9,r'$\sigma^2/n(z=0.1)$, 2 yr',color='green')
ax2.text(0.08,sigv[2]**2/twoyeardensity+1e10,r'$\sigma^2/n(z=0.2)$, 2 yr',color='red')

ax2.plot([1./(zs[0]*3e3*2),0.1],[sigv[0]**2/tenyeardensity,sigv[0]**2/tenyeardensity],color='blue')
ax2.plot([1./(zs[1]*3e3*2),0.1],[sigv[1]**2/tenyeardensity,sigv[1]**2/tenyeardensity],color='green')
ax2.plot([1./(zs[2]*3e3*2),0.1],[sigv[2]**2/tenyeardensity,sigv[2]**2/tenyeardensity],color='red')
ax2.text(0.0007,sigv[0]**2/tenyeardensity-2.5e8,r'$\sigma^2/n(z=0.05)$, 10 yr',color='blue')
ax2.text(0.0007,sigv[1]**2/tenyeardensity-.8e9,r'$\sigma^2/n(z=0.1)$, 10 yr',color='green')
ax2.text(0.0007,sigv[2]**2/tenyeardensity-3e9,r'$\sigma^2/n(z=0.2)$, 10 yr',color='red')

ax2.set_xlim((5e-4,.5))
ax2.set_ylim((1e8,2e13))
ax2.set_yscale("log", nonposx='clip')
ax2.set_xscale("log", nonposx='clip')
ax2.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
ax2.set_ylabel(r'[km$^{2}$ s$^{-2}$ Mpc$^{3}$ $h^{-3}$]')
# ax2.legend()
plt.tight_layout()
plt.savefig('noise.png')


# transfer = numpy.loadtxt("my_transfer_out.dat")

# plt.plot(transfer[:,0],transfer[:,1])
# plt.yscale("log", nonposx='clip')
# plt.show()
