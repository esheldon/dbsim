import logging
import numpy as np
from pprint import pprint

import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation
from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import log_pars, TryAgainError, Namer

import minimof

logger = logging.getLogger(__name__)

class FitterBase(dict):
    def __init__(self, run_conf, nband, rng):

        self.nband=nband
        self.rng=rng
        self.update(run_conf)

    def go(self, mbobs_list):
        """
        do measurements.  This is abstract
        """
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep

        ppars=conf['priors']

        # g
        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])
        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp=ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if conf['model']=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            assert fp['type'] == 'normal','only normal prior supported for fracdev'

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                T_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:

            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                T_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        ptype=ppars['type']

        if ptype=="flat":
            prior=ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            prior=ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype=='normal':
            prior = ngmix.priors.Normal(
                ppars['cen'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='normal2d':
            prior=ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='ba':
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior



     

class MOFFitter(FitterBase):
    def __init__(self, *args, **kw):

        super(MOFFitter,self).__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])


    def go(self, mbobs_list):
        """
        run the multi object fitter
        """

        self._fit_all_psfs(mbobs_list, self['mof']['psf'])

        mofc = self['mof']
        fitter = minimof.MOFStamps(
            mbobs_list,
            mofc['model'],
            prior=self.mof_prior,
        )
        guess=minimof.mof.get_stamp_guesses(
            mbobs_list,
            mofc['detband'],
            mofc['model'],
            self.rng,
        )
        fitter.go(guess)

        res=fitter.get_result()

        if res['flags'] != 0:
            data=None
        else:
            reslist=fitter.get_result_list()
            data=self._get_output(reslist)

        return data

    def _fit_all_psfs(self, mbobs_list, psf_conf):
        fitter=AllPSFFitter(mbobs_list, psf_conf)
        fitter.go()

    def _get_dtype(self, npars):
        n=Namer(front=self['mof']['model'])
        return [
            ('psf_g','f8',2),
            ('psf_T','f8'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
        ]

    def _get_output(self,reslist):

        npars=reslist[0]['pars'].size

        n=Namer(front=self['mof']['model'])

        dt=self._get_dtype(npars)
        output=np.zeros(len(reslist), dtype=dt)

        for i,res in enumerate(reslist):
            t=output[i] 

            for name,val in res.items():
                if 'psf' in name:
                    t[name] = val
                else:
                    nname=n(name)
                    t[nname] = val

        return output
            
class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list=mbobs_list
        self.psf_conf=psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    self._fit_one(psf_obs)

    def _fit_one(self, obs):
        Tguess=4.0*obs.jacobian.get_scale()**2

        pconf=self.psf_conf

        runner=ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )
        runner.go(ntry=pconf['ntry'])

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()
        obs.update_meta_data({'fitter':psf_fitter})

        if res['flags']==0:
            gmix=psf_fitter.get_gmix()
            obs.set_gmix(gmix)
        else:
            raise BootPSFFailure("failed to fit psfs: %s" % str(res))
