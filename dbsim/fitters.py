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

        if 'priors' not in conf:
            return None

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
                ppars['mean'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='log-normal':
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
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


    def go(self, mbobs_list, ntry=2, get_fitter=False):
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

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])

            mofc = self['mof']
            fitter = mof.MOFStamps(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
            )
            for i in range(ntry):
                guess=mof.moflib.get_stamp_guesses(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            average_fof_shapes = self.get('average_fof_shapes',False)
            if average_fof_shapes:
                logger.debug('averaging fof shapes')
                resavg=fitter.get_result_averaged_shapes()
                data=self._get_output([resavg], fitter.nband)
            else:
                reslist=fitter.get_result_list()
                data=self._get_output(reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data


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

class MOFFitterFull(MOFFitter):
    def __init__(self, *args, **kw):
        """
        we don't use the MOFFitter init
        """
        FitterBase.__init__(self, *args, **kw)

    def go(self, mbobs, cat, ntry=2, get_fitter=False):
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

        mofc = self['mof']
        nband=len(mbobs)
        jacobian=mbobs[0][0].jacobian

        prior=self._get_prior(
            cat,
            jacobian,
        )

        try:
            _fit_all_psfs([mbobs], mofc['psf'])

            fitter = mof.MOF(
                mbobs,
                mofc['model'],
                cat.size,
                prior=prior,
            )

            for i in range(ntry):
                guess=mof.moflib.get_full_image_guesses(
                    cat,
                    nband,
                    jacobian,
                    mofc['model'],
                    self.rng,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            average_fof_shapes = self.get('average_fof_shapes',False)
            if average_fof_shapes:
                raise NotImplementedError('make sure works for full mof')
                logger.debug('averaging fof shapes')
                resavg=fitter.get_result_averaged_shapes()
                data=self._get_output([resavg], fitter.nband)
            else:
                reslist=fitter.get_result_list()
                data=self._get_output(reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data

    def _get_prior(self, objects, jacobian):
        """
        Note a single jacobian is being sent.  for multi-band this
        is the same as assuming they are all on the same coordinate system.
        
        assuming all images have the 
        prior for N objects.  The priors are the same for
        structural parameters, the only difference being the
        centers
        """

        conf=self['mof']
        ppars=conf['priors']

        nobj=len(objects)

        cen_priors=[]

        cen_sigma=jacobian.get_scale() # a pixel
        for i in range(nobj):
            row=objects['y'][i]#-1
            col=objects['x'][i]#-1

            v, u = jacobian(row, col)
            p=ngmix.priors.CenPrior(
                v,
                u,
                cen_sigma, cen_sigma,
                rng=self.rng,
            )
            cen_priors.append(p)

        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])
        F_prior = self._get_prior_generic(ppars['flux'])

        if conf['model']=='bdf':
            fracdev_prior = ngmix.priors.Normal(0.5, 0.1, rng=self.rng)

            return mof.priors.PriorBDFSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                fracdev_prior,
                [F_prior]*self.nband,
            )
        else:
            return mof.priors.PriorSimpleSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                [F_prior]*self.nband,
            )


class MetacalFitter(FitterBase):
    """
    run metacal on all objects found in the image, using
    the deblended or "corrected" images produced by the
    multi-object fitter
    """
    def __init__(self, *args, **kw):

        self.mof_fitter=kw.pop('mof_fitter',None)

        super(MetacalFitter,self).__init__(*args, **kw)

        self.metacal_prior = self._get_prior(self['metacal'])
        self['metacal']['symmetrize_weight'] = \
            self['metacal'].get('symmetrize_weight',False)


    def go(self, mbobs_list_input):
        """
        do all fits and return fitter, data

        metacal data are appended to the mof data for each object
        """

        if self.mof_fitter is not None:
            # for mof fitting, we expect a list of mbobs_lists
            fitter, mof_data = self.mof_fitter.go(
                mbobs_list_input,
                get_fitter=True,
            )
            if mof_data is None:
                return None

            # this gets all objects, all bands in a list of MultiBandObsList
            mbobs_list = fitter.make_corrected_obs()

            if False:
                self._show_corrected_obs(mbobs_list_input, mbobs_list)
        else:
            mbobs_list = mbobs_list_input
            mof_data=None

        if self['metacal']['symmetrize_weight']:
            self._symmetrize_weights(mbobs_list)

        return self._do_all_metacal(mbobs_list, data=mof_data)

    def _symmetrize_weights(self, mbobs_list):
        for mbobs in mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    self._symmetrize_weight(obs.weight)

        if False:
            from . import visualize
            visualize.view_mbobs_list(mbobs_list,title='symmetrized',weight=True)
            if 'q'==input('hit a key (q to quit): '):
                stop

    def _symmetrize_weight(self, wt):
        """
        symmetrize raw weight pixels in all of the maps
        """
        assert wt.shape[0] == wt.shape[1]

        for k in (1,2,3):
            wt_rot = np.rot90(wt, k=k)
            wzero  = np.where(wt_rot == 0.0)

            if wzero[0].size > 0:
                wt[wzero] = 0.0

    def _show_corrected_obs(self, mbobs_list, corrected_mbobs_list):
        import images

        if len(mbobs_list[0])==3:
            for i,mbobs in enumerate(corrected_mbobs_list):
                bim0=mbobs_list[i][0][0].image.transpose()
                gim0=mbobs_list[i][1][0].image.transpose()
                rim0=mbobs_list[i][2][0].image.transpose()
                bim=mbobs[0][0].image.transpose()
                gim=mbobs[1][0].image.transpose()
                rim=mbobs[2][0].image.transpose()
                mval=max(bim.max(), gim.max(), rim.max())
                rgb0=images.get_color_image(rim0/mval, gim0/mval, bim0/mval, nonlinear=0.1)
                rgb=images.get_color_image(rim/mval, gim/mval, bim/mval, nonlinear=0.1)
                #images.view(mbobs[0][0].image,title='%d' % i)
                imlist=[
                    rgb0/rgb0.max(), rgb/rgb.max(),
                    mbobs_list[i][0][0].weight, mbobs[0][0].weight,
                ]
                titles=['orig','corrected','weight orig','weight corr']
                images.view_mosaic(imlist, titles=titles)
        else:
            for i,mbobs in enumerate(corrected_mbobs_list):
                im0=mbobs_list[i][0][0].image
                im=mbobs[0][0].image
                #images.view(mbobs[0][0].image,title='%d' % i)
                imlist=[
                    im0, im, 
                    mbobs_list[i][0][0].weight, mbobs[0][0].weight,
                ]
                titles=['orig','corrected','weight orig','weight corr']
                images.view_mosaic(imlist, titles=titles)

        if 'q'==input('hit a key (q to quit): '):
            stop


    def _do_all_metacal(self, mbobs_list, data=None):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        nband=len(mbobs_list[0])

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):
            if self._check_flags(mbobs):
                try:
                    boot=self._do_one_metacal(mbobs)
                    res=boot.get_metacal_result()
                except (BootPSFFailure, BootGalFailure) as err:
                    logger.debug(str(err))
                    res={'mcal_flags':1}
                except RuntimeError as err:
                    # argh galsim and its generic errors
                    logger.info('caught RuntimeError: %s' % str(err))
                    res={'mcal_flags':1}

                if res['mcal_flags'] != 0:
                    logger.debug("        metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_metacal_output(res, nband, mbobs)
                    if data is not None:
                        odata = data[i:i+1]
                        fit_data = eu.numpy_util.add_fields(
                            fit_data,
                            odata.dtype.descr,
                        )
                        eu.numpy_util.copy_fields(odata, fit_data)

                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output


    def _do_one_metacal(self, mbobs):
        conf=self['metacal']

        psf_pars=conf['psf']
        max_conf=conf['max_pars']

        tpsf_obs=mbobs[0][0].psf
        if not tpsf_obs.has_gmix():
            _fit_one_psf(tpsf_obs, psf_pars)

        psf_Tguess=tpsf_obs.gmix.get_T()

        boot=self._get_bootstrapper(mbobs)
        if 'lm_pars' in psf_pars:
            psf_fit_pars=psf_pars['lm_pars']
        else:
            psf_fit_pars=None

        boot.fit_metacal(

            psf_pars['model'],

            conf['model'],
            max_conf['pars'],

            psf_Tguess,
            psf_fit_pars=psf_fit_pars,
            psf_ntry=psf_pars['ntry'],

            prior=self.metacal_prior,
            ntry=max_conf['ntry'],

            metacal_pars=conf['metacal_pars'],
        )
        return boot

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags=self['metacal'].get('bmask_flags',None)

        isok=True
        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    w=np.where( (obs.bmask & flags) != 0 )
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


    def _print_result(self, data):
        mess="        mcal s2n: %g Trat: %g"
        logger.debug(mess % (data['mcal_s2n'][0], data['mcal_T_ratio'][0]))

    def _get_metacal_dtype(self, npars, nband):
        dt=[
            ('image_id','i4'),
            ('x','f8'),
            ('y','f8'),
        ]
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
                (n('s2n_r'),'f8'),
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

    def _get_metacal_output(self, allres, nband, mbobs):
        # assume one epoch and line up in all
        # bands
        assert len(mbobs[0])==1,'one epoch only'



        npars=len(allres['noshear']['pars'])
        dt = self._get_metacal_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        data0=data[0]
        data0['y'] = mbobs[0][0].meta['orig_row']
        data0['x'] = mbobs[0][0].meta['orig_col']

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

class MetacalAvgFitter(MetacalFitter):
    def _do_one_metacal(self, mbobs):
        nrand = self['metacal']['nrand']

        reslist=[]
        first=True
        for i in range(nrand):
            try:
                tboot=super(MetacalAvgFitter,self)._do_one_metacal(mbobs)

                res=tboot.get_metacal_result()
                reslist.append(res)
                if first:
                    boot=tboot
                    first=False

            except (BootPSFFailure, BootGalFailure) as err:
                logger.debug(str(err))
                res={'mcal_flags':1}
            except RuntimeError as err:
                # argh galsim and its generic errors
                logger.info('caught RuntimeError: %s' % str(err))
                res={'mcal_flags':1}

        if len(reslist)==0:
            raise BootGalFailure('none of the metacal fits worked')

        # this is the first, corresponds to the result in the
        # boot variable

        types=self['metacal']['metacal_pars']['types']
        dontavg=[
            'flags','nfev','ier','errmsg','model',
            'npix','lnprob','chi2per','dof','ntry',
        ]
        res=reslist[0]
        nkept = len(reslist)
        if nkept > 1:
            for tres in reslist[1:]:
                for type in types:
                    typeres=res[type]
                    ttyperes=tres[type]
                    for key in typeres:
                        if key not in dontavg:
                            #print('    key:',key)
                            typeres[key] += ttyperes[key]

            fac = 1.0/nkept
            for type in types:
                typeres=res[type]
                for key in typeres:
                    if key not in dontavg:
                        typeres[key] *= fac
            
        return boot

class AdmomMetacalFitter(MetacalFitter):
    #def __init__(self, *args, **kw):
    #    super(AdmomMetacalFitter,self).__init__(*args, **kw)
        
    def _do_one_metacal(self, mbobs):
        conf=self['metacal']

        boot=self._get_bootstrapper(mbobs)

        psf_Tguess=4.0*mbobs[0][0].jacobian.get_scale()**2

        boot.fit_metacal(
            psf_Tguess=psf_Tguess,
        )
        return boot

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import AdmomMetacalBootstrapper

        return AdmomMetacalBootstrapper(
            mbobs,
            admom_pars=self['metacal'].get('admom_pars',None),
            metacal_pars=self['metacal']['metacal_pars'],
        )

class AdmomMetacalAvgFitter(AdmomMetacalFitter):
    def _do_one_metacal(self, mbobs):
        nrand = self['metacal']['nrand']

        reslist=[]
        first=True
        for i in range(nrand):
            try:
                tboot=super(AdmomMetacalAvgFitter,self)._do_one_metacal(mbobs)

                res=tboot.get_metacal_result()
                reslist.append(res)
                if first:
                    boot=tboot
                    first=False

            except (BootPSFFailure, BootGalFailure) as err:
                logger.debug(str(err))
                res={'mcal_flags':1}
            except RuntimeError as err:
                # argh galsim and its generic errors
                logger.info('caught RuntimeError: %s' % str(err))
                res={'mcal_flags':1}

        if len(reslist)==0:
            raise BootGalFailure('none of the metacal fits worked')

        # this is the first, corresponds to the result in the
        # boot variable

        types=self['metacal']['metacal_pars']['types']
        dontavg=[
            'flags','model',
            'npix','ntry',
            'numiter','nimage','flagstr',
        ]
        res=reslist[0]
        nkept = len(reslist)
        if nkept > 1:
            for tres in reslist[1:]:
                for type in types:
                    typeres=res[type]
                    ttyperes=tres[type]
                    for key in typeres:
                        if key not in dontavg:
                            #print('    key:',key)
                            typeres[key] += ttyperes[key]

            fac = 1.0/nkept
            for type in types:
                typeres=res[type]
                for key in typeres:
                    if key not in dontavg:
                        typeres[key] *= fac
            
        return boot


class MomentMetacalFitter(MetacalFitter):
    def __init__(self, *args, **kw):
        super(MomentMetacalFitter,self).__init__(*args, **kw)
        self._set_mompars()
        
    def _set_mompars(self):
        wpars=self['weight']

        T=ngmix.moments.fwhm_to_T(wpars['fwhm'])

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight=ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm=weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight=weight

        wpars['use_canonical_center']=wpars.get('use_canonical_center',False)

    def _do_one_metacal(self, mbobs):
        assert len(mbobs)==1
        assert len(mbobs[0])==1

        conf=self['metacal']

        mpars=conf['metacal_pars']

        odict=ngmix.metacal.get_all_metacal(
            mbobs,
            rng=self.rng,
            **mpars
        )


        res={}

        for type in mpars['types']:
            mbobs=odict[type]
            obs=mbobs[0][0]

            tres=self._measure_moments(obs)
            tres['g'] = tres['e']
            tres['g_cov'] = tres['e_cov']

            if type=='noshear':
                pres  = self._measure_moments(obs.psf)
                tres['gpsf'] = pres['e']
                tres['Tpsf'] = pres['T']

            res[type]=tres

        res['mcal_flags']=0
        boot=MomentBootstrapperFaker(res)
        return boot 



    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import AdmomMetacalBootstrapper

        return AdmomMetacalBootstrapper(
            mbobs,
            admom_pars=self['metacal'].get('admom_pars',None),
            metacal_pars=self['metacal']['metacal_pars'],
        )


    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        wpars=self['weight']

        if wpars['use_canonical_center']:
            #logger.debug('        getting moms with canonical center')
        
            ccen=(numpy.array(obs.image.shape)-1.0)/2.0
            jold=obs.jacobian
            obs.jacobian = ngmix.Jacobian(
                row=ccen[0],
                col=ccen[1],
                dvdrow=jold.dvdrow,
                dudrow=jold.dudrow,
                dvdcol=jold.dvdcol,
                dudcol=jold.dudcol,

            )

        res = self.weight.get_weighted_moments(obs=obs,maxrad=1.e9)

        if wpars['use_canonical_center']:
            obs.jacobian=jold

        if res['flags'] != 0:
            raise BootGalFailure("        moments failed")

        res['numiter'] = 1

        return res

class MomentBootstrapperFaker(object):
    def __init__(self, res):
        self.res=res
    def get_metacal_result(self):
        return self.res

class MaxFitter(FitterBase):
    """
    run a max like fitter
    """
    def __init__(self, *args, **kw):

        self.mof_fitter=kw.pop('mof_fitter',None)

        super(MaxFitter,self).__init__(*args, **kw)

        self.prior = self._get_prior(self['max'])


    def go(self, mbobs_list_input):
        """
        do all fits and return fitter, data

        metacal data are appended to the mof data for each object
        """

        if self.mof_fitter is not None:
            # for mof fitting, we expect a list of mbobs_lists
            fitter, mof_data = self.mof_fitter.go(
                mbobs_list_input,
                get_fitter=True,
            )
            if mof_data is None:
                return None

            # this gets all objects, all bands in a list of MultiBandObsList
            mbobs_list = fitter.make_corrected_obs()

        else:
            mbobs_list = mbobs_list_input
            mof_data=None

        return self._do_all_fits(mbobs_list, data=mof_data)


    def _do_all_fits(self, mbobs_list, data=None):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        nband=len(mbobs_list[0])

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):
            if self._check_flags(mbobs):
                try:
                    boot, pres=self._do_one_fit(mbobs)
                    res=boot.get_max_fitter().get_result()
                except (BootPSFFailure, BootGalFailure):
                    res={'flags':1}

                if res['flags'] != 0:
                    logger.debug("        metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_output(res, pres, nband)
                    if data is not None:
                        odata = data[i:i+1]
                        fit_data = eu.numpy_util.add_fields(
                            fit_data,
                            odata.dtype.descr,
                        )
                        eu.numpy_util.copy_fields(odata, fit_data)

                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output


    def _do_one_fit(self, mbobs):
        conf=self['max']

        psf_pars=conf['psf']
        max_conf=conf['max_pars']

        tpsf_obs=mbobs[0][0].psf
        if not tpsf_obs.has_gmix():
            _fit_one_psf(tpsf_obs, psf_pars)

        psf_Tguess=tpsf_obs.gmix.get_T()

        boot=self._get_bootstrapper(mbobs)

        boot.fit_psfs(
            psf_pars['model'],
            psf_Tguess,
            fit_pars=psf_pars['lm_pars'],
            ntry=psf_pars['ntry'],
        )
        boot.fit_max(

            conf['model'],
            max_conf['pars'],

            prior=self.prior,
            ntry=max_conf['ntry'],
        )

        pres=self._get_object_psf_stats(boot.mb_obs_list)
        return boot, pres

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags=self['max'].get('bmask_flags',None)

        isok=True
        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    w=np.where( (obs.bmask & flags) != 0 )
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


    def _print_result(self, data):
        n=self._get_namer()
        mess="        s2n: %g Trat: %g"
        logger.debug(mess % (data[n('s2n')][0], data[n('T_ratio')][0]))

    def _get_namer(self):
        model=self['max']['model']
        return Namer(front=model)

    def _get_dtype(self, npars, nband):
        dt=[]

        n=self._get_namer()
        dt += [
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

        return dt

    def _get_output(self, res, pres, nband):
        npars=len(res['pars'])
        dt = self._get_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        n=self._get_namer()

        data0=data[0]

        data0['psf_g'] = pres['g']
        data0['psf_T'] = pres['T']

        for name in res:
            nn=n(name)
            if nn in data.dtype.names:
                data0[nn] = res[name]

        # this relies on noshear coming first in the metacal
        # types
        data0[n('T_ratio')] = data0[n('T')]/data0['psf_T']

        return data

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import Bootstrapper

        return Bootstrapper(
            mbobs,
            verbose=False,
        )

    def _get_object_psf_stats(self, mbobs):
        """
        get the s/n for the given object.  This uses just the model
        to calculate the s/n, but does use the full weight map
        """
        g1sum=0.0
        g2sum=0.0
        Tsum=0.0
        wsum=0.0

        for band,obslist in enumerate(mbobs):
            for obsnum,obs in enumerate(obslist):
                twsum=obs.weight.sum()
                wsum += twsum

                tg1, tg2, tT = obs.psf.gmix.get_g1g2T()

                g1sum += tg1*twsum
                g2sum += tg2*twsum
                Tsum += tT*twsum

        g1 = g1sum/wsum
        g2 = g2sum/wsum
        T = Tsum/wsum

        return {
            'g':[g1,g2],
            'T':T,
        }



def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()
     


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list=mbobs_list
        self.psf_conf=psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    _fit_one_psf(psf_obs, self.psf_conf)

def _fit_one_psf(obs, pconf):
    Tguess=4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss=ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner=ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )


    else:
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
