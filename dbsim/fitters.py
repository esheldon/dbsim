import logging
import numpy as np

import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation
from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import log_pars, TryAgainError

import minimof

logger = logging.getLogger(__name__)

class FitterBase(dict):
    def __init__(self, run_conf, nband, rng):

        self.sim=sim
        self.nband=nband
        self.rng=rng
        self.update(run_conf)

        self.mof_prior = self._get_prior(self['mof_priors'])
        self.metacal_prior = self._get_prior(self['metacal_priors'])

    def go(self, mbobs_list):
        """
        do measurements.  This is abstract
        """
        raise NotImplementedError("implement go()")

    def _get_prior(self, ppars):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep

        # g
        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = ngmix.priors.GPriorBA(gp['sigma'], rng=self.rng)

        # T
        Tp = ppars['T']

        if Tp['type']=="flat":
            T_prior=ngmix.priors.FlatPrior(*Tp['pars'], rng=self.rng)
        elif Tp['type'] in ['TwoSidedErf',"two-sided-erf"]:
            T_prior_pars = Tp['pars']
            T_prior=ngmix.priors.TwoSidedErf(*T_prior_pars, rng=self.rng)
        else:
            raise ValueError("bad Tprior: '%s'" % Tp['type'])

        # flux
        Fp=ppars['flux']

        if Fp['type'] in ['TwoSidedErf',"two-sided-erf"]:
            flux_prior=ngmix.priors.TwoSidedErf(*Fp['pars'], rng=self.rng)

        elif Fp['type']=="flat":
            flux_prior=ngmix.priors.FlatPrior(*Fp['pars'], rng=self.rng)

        else:
            raise ValueError("bad flux prior: '%s'" % Fp['type'])

        # center
        cp=ppars['cen']
        assert cp['type'] == 'normal2d'
        fit_cen_sigma=cp['sigma']
        cen_prior=ngmix.priors.CenPrior(
            0.0,
            0.0,
            fit_cen_sigma,
            fit_cen_sigma,
            rng=self.rng,
        )

        if self['fit_model']=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            assert fp['type'] == 'normal','only normal prior supported for fracdev'
            
            fracdev_prior = ngmix.priors.Normal(
                fp['cen'],
                fp['sigma'],
                rng=rng,
            )

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



class MOFFItter(FitterBase):
    pass

class MOFFitter(FitterBase):
    def go(self, mbobs_list):
        """
        run the multi object fitter
        """
        fitter = minimof.MOFStamps(
            mbobs_list,
            run_conf['fit_model'],
            prior=mof_prior,
        )
        guess=minimof.mof.get_stamp_guesses(
            mbobs_list,
            self['detband'],
            self['fit_model'],
            self.rng,
        )
        fitter.go(guess)

        res=fitter.get_result()
        return None
        if res['flags'] != 0:
            data=None
        else:
            data=self._get_output(res)

        return data

