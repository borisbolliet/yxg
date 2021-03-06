# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyccl as ccl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

cosmo=ccl.Cosmology(Omega_c=0.26066676,
                    Omega_b=0.048974682,
                    h=0.6766,
                    sigma8=0.8102,
                    n_s=0.9665,
                    mass_function='tinker')

def get_corr(cv):
    return cv/np.sqrt(np.diag(cv)[:,None]*np.diag(cv)[None,:])

def plot_covar(predir,sample,bn,sample_label,kmax=1.,lmin=0.):
    z,nz=np.loadtxt('data/dndz/'+sample.upper()+'_bin%d.txt'%bn,
                    unpack=True)
    zmean=np.sum(z*nz)/np.sum(nz)
    chi=ccl.comoving_radial_distance(cosmo,1/(1+zmean))
    lmax=kmax*chi-0.5

    if sample!="2mpz":
        gsample=sample+'%d'%bn
    else:
        gsample=sample
    l = np.load("output_default/cls_"+gsample+'_'+gsample+".npz")['ls']
    msk_gg=(l<=lmax) & (l>=lmin)
    msk_gy=(l<=lmax)
    ngg=np.sum(msk_gg)
    ngy=np.sum(msk_gy)
    nt=ngg+ngy
    cv_gg_gg=np.load(predir+'/cov_comb_m_'+
                     gsample+'_'+gsample+'_'+
                     gsample+'_'+gsample+'.npz')['cov'][msk_gg,:][:,msk_gg]
    cv_gg_gy=np.load(predir+'/cov_comb_m_'+
                     gsample+'_'+gsample+'_'+
                     gsample+'_y_milca.npz')['cov'][msk_gg,:][:,msk_gy]
    cv_gy_gy=np.load(predir+'/cov_comb_m_'+
                     gsample+'_y_milca_'+
                     gsample+'_y_milca.npz')['cov'][msk_gy,:][:,msk_gy]
    cov=np.zeros([nt,nt])
    cov[:ngg,:ngg]=cv_gg_gg
    cov[:ngg,ngg:]=cv_gg_gy
    cov[ngg:,:ngg]=cv_gg_gy.T
    cov[ngg:,ngg:]=cv_gy_gy

    plt.figure()
    ax=plt.gca()
    im=ax.imshow(get_corr(cov),interpolation='nearest',
                 vmin=0,vmax=1,cmap=cm.gray)
    ax.tick_params(length=0)
    ax.set_xticks([ngg/2,ngg+ngy/2])
    ax.set_xticklabels(['$g\\,\\times\\,g$','$g\\,\\times\\,y$'])
    ax.set_yticks([0.4*ngg,ngg+0.4*ngy])
    ax.set_yticklabels(['$g\\,\\times\\,g$','$g\\,\\times\\,y$'],
                       rotation=90)
    ax.text(0.03,0.04,sample_label,transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1),fontsize=14)
    ax.tick_params(labelsize="x-large")
    plt.savefig('notes/paper/cov_'+gsample+'.pdf',
                bbox_inches='tight')

plot_covar("output_default","2mpz",1,'2MPZ',lmin=0.)
for b in range(5):
    plot_covar("output_default","wisc",b+1,'WI$\\times$SC-%d'%(b+1),lmin=10.)
plt.show()
