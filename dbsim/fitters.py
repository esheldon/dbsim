import logging
import numpy as np
from pprint import pprint

import esutil as eu
import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation
from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import log_pars, TryAgainError, Namer

import mof

logger = logging.getLogger(__name__)

METACAL_TYPES=['noshear','1p','1m','2p','2m']

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


    def go(self, mbobs_list, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object

        returns
        -------
        data: ndarray
            Array with all output fields
        """

        self._fit_all_psfs(mbobs_list, self['mof']['psf'])

        mofc = self['mof']
        fitter = mof.MOFStamps(
            mbobs_list,
            mofc['model'],
            prior=self.mof_prior,
        )
        guess=mof.moflib.get_stamp_guesses(
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
            data=self._get_output(reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data


    def _fit_all_psfs(self, mbobs_list, psf_conf):
        fitter=AllPSFFitter(mbobs_list, psf_conf)
        fitter.go()

    def _get_dtype(self, npars, nband):
        n=Namer(front=self['mof']['model'])
        dt = [
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
            (n('flux'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

    def _get_output(self,reslist,nband):

        npars=reslist[0]['pars'].size

        model=self['mof']['model']
        n=Namer(front=model)

        dt=self._get_dtype(npars, nband)
        output=np.zeros(len(reslist), dtype=dt)

        for i,res in enumerate(reslist):
            t=output[i] 

            for name,val in res.items():
                if name=='nband':
                    continue

                if 'psf' in name:
                    t[name] = val
                else:
                    nname=n(name)
                    t[nname] = val

        return output
            
class MetacalFitter(MOFFitter):
    """
    run metacal on all objects found in the image, using
    the deblended or "corrected" images produced by the
    multi-object fitter
    """
    def __init__(self, *args, **kw):

        super(MetacalFitter,self).__init__(*args, **kw)

        self.metacal_prior = self._get_prior(self['metacal'])

    def go(self, mbobs_list):
        """
        do all fits and return fitter, data

        metacal data are appended to the mof data for each object
        """
        mof_fitter, data = super(MetacalFitter,self).go(
            mbobs_list,
            get_fitter=True,
        )
        if data is None:
            return None

        # this gets all objects, all bands in a list of MultiBandObsList
        corrected_mbobs_list = mof_fitter.make_corrected_obs()
        return self._do_metacal(corrected_mbobs_list, data)

    def _do_metacal(self, mbobs_list, data):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        nband=len(mbobs_list[0])

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):
            try:
                boot=self._do_one_metacal(mbobs, data[i])
                res=boot.get_metacal_result()
            except (BootPSFFailure, BootGalFailure):
                res={'mcal_flags':1}

            if res['mcal_flags'] != 0:
                logger.debug("        metacal fit failed")
            else:
                # make sure we send an array
                odata = data[i:i+1]
                new_data = self._get_metacal_output(odata, res, nband)
                self._print_result(new_data)
                datalist.append(new_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output

    def _do_one_metacal(self, mbobs, data):
        conf=self['metacal']

        psf_pars=conf['psf']
        max_conf=conf['max_pars']

        psf_Tguess=mbobs[0][0].psf.gmix.get_T()

        boot=self._get_bootstrapper(mbobs)
        boot.fit_metacal(

            psf_pars['model'],

            conf['model'],
            max_conf['pars'],

            psf_Tguess,
            psf_fit_pars=psf_pars['lm_pars'],
            psf_ntry=psf_pars['ntry'],

            prior=self.metacal_prior,
            ntry=max_conf['ntry'],

            metacal_pars=conf['metacal_pars'],
        )
        return boot

    def _print_result(self, data):
        mess="        mcal s2n: %g Trat: %g"
        logger.debug(mess % (data['mcal_s2n'][0], data['mcal_T_ratio'][0]))

    def _get_metacal_dtype(self, npars, nband):
        dt=[]
        for mtype in METACAL_TYPES:
            if mtype == 'noshear':
                back=None
            else:
                back=mtype

            n=Namer(front='mcal', back=back)
            if mtype=='noshear':
                dt += [
                    (n('psf_g'),'f8',2),
                    (n('psf_T'),'f8'),
                ]

            dt += [
                (n('nfev'),'i4'),
                (n('s2n'),'f8'),
                (n('pars'),'f8',npars),
                (n('pars_cov'),'f8',(npars,npars)),
                (n('g'),'f8',2),
                (n('g_cov'),'f8',(2,2)),
                (n('T'),'f8'),
                (n('T_err'),'f8'),
                (n('T_ratio'),'f8'),
                (n('flux'),'f8',nband),
                (n('flux_cov'),'f8',(nband,nband)),
                (n('flux_err'),'f8',nband),
            ]

        return dt

    def _get_metacal_output(self, odata, allres, nband):
        npars=len(allres['noshear']['pars'])
        extra_dt = self._get_metacal_dtype(npars, nband)
        data = eu.numpy_util.add_fields(odata, extra_dt)
        data0=data[0]

        for mtype in METACAL_TYPES:

            if mtype == 'noshear':
                back=None
            else:
                back=mtype

            n=Namer(front='mcal', back=back)

            res=allres[mtype]

            if mtype=='noshear':
                data0[n('psf_g')] = res['gpsf']
                data0[n('psf_T')] = res['Tpsf']

            for name in res:
                nn=n(name)
                if nn in data.dtype.names:
                    data0[nn] = res[name]

            # this relies on noshear coming first in the metacal
            # types
            data0[n('T_ratio')] = data0[n('T')]/data0['mcal_psf_T']

        return data

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import MaxMetacalBootstrapper

        return MaxMetacalBootstrapper(
            mbobs,
            verbose=False,
        )


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
