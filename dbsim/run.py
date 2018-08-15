import os
import time
import numpy as np
import logging
import fitsio
import esutil as eu
import ngmix
from . import simulation
from . import fitters

logger = logging.getLogger(__name__)

def go(sim_conf,
       fit_conf,
       ntrials,
       seed,
       output_file,
       show=False,
       make_plots=False):
    """
    run the simulation and fitter
    """
    rng = np.random.RandomState(seed)
    fitseed=rng.randint(0,2**30)
    fitrng = np.random.RandomState(fitseed)

    sim=simulation.Sim(sim_conf, rng)

    fitter=get_fitter(sim_conf, fit_conf, fitrng)

    nsim=0
    nfit=0
    nobj_detected=0
    tm0_main = time.time()
    tm_sim=0.0
    tm_fit=0.0

    fit_conf['meta']=fit_conf.get('meta',None)
    metad=fit_conf['meta']

    datalist=[]
    for i in range(ntrials):

        logger.debug("trial: %d/%d" % (i+1,ntrials))

        tm0=time.time()
        logger.debug('simulating images')
        sim.make_obs()
        tm_sim += time.time()-tm0

        nsim += 1

        if show:
            sim.show()
            if 'q'==input('hit a key (q to quit): '):
                return

        if metad is not None and metad['dometa']:
            resdict, nobj, tm = do_meta(sim, fit_conf, fitter)
            reslist=[resdict]
        else:
            reslist, nobj, tm = do_fits(sim, fit_conf, fitter, show=show)

        if reslist is not None:
            nobj_detected += nobj
            tm_fit += tm
            datalist += reslist
            if nobj > 0:
                nfit += 1


    elapsed_time=time.time()-tm0_main
    nkept = len(datalist)

    meta=make_meta(
        ntrials, nsim, nfit, nobj_detected,
        nkept, elapsed_time, tm_sim, tm_fit,
    )

    logger.info("kept: %d/%d %.2f" % (nkept,ntrials,float(nkept)/ntrials))
    logger.info("time minutes: %g" % meta['tm_minutes'][0])
    logger.info("time per trial: %g" % meta['tm_per_trial'][0])
    logger.info("time per sim: %g" % meta['tm_per_sim'][0])
    if nfit > 0:
        logger.info("time per fit: %g" % meta['tm_per_fit'][0])
        logger.info("time fit per detected object: %g" % meta['tm_per_obj_detected'][0])

    if nkept == 0:
        logger.info("no results to write")
    else:
        if metad is not None and metad['dometa']:
            write_meta(output_file, datalist, meta, fit_conf)
        else:
            data = eu.numpy_util.combine_arrlist(datalist)
            write_output(output_file, data, meta)


def do_meta(sim, fit_conf, fitter, show=False):
    """
    currently we only do the full version, making
    metacal images for the full image set and
    sending all to MOF 
    """
    mtype=fit_conf['meta']['type']
    if mtype=='meta-detect':
        tup = do_meta_detect(sim, fit_conf, fitter, show=False)
    elif mtype in ['meta-mof','meta-max']:
        tup = do_meta_mof(sim, fit_conf, fitter, show=False)
    else:
        raise ValueError("bad meta type: '%s'" % mtype)
    
    return tup

def do_meta_detect(sim, fit_conf, fitter, show=False):
    """
    metacal the entire process, including detection.
    This means you lose a lot of detections
    """
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        **metacal_pars
    )

    nobj=0
    tm_fit=0.0
    reslists={}
    for key in odict:
        reslist, tnobj, ttm = do_fits(
            sim,
            fit_conf,
            fitter,
            obs=odict[key],
            show=show,
        )
        nobj = max(nobj, tnobj)
        tm_fit += ttm
        reslists[key] = reslist

    return reslists, nobj, tm_fit

def do_meta_mof(sim, fit_conf, fitter, show=False):
    """
    metacal the MOF process but not detection

    also can do without mof

    build the catalog based on original images, but then
    run MOF on sheared versions
    """

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        **metacal_pars
    )

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medsifier=sim.get_medsifier()

    nobj=0
    tm_fit=0.0
    reslists={}
    for key in odict:
        reslist, tnobj, ttm = do_fits(
            sim,
            fit_conf,
            fitter,
            cat=medsifier.cat,
            seg=medsifier.seg,
            obs=odict[key],
            show=show,
        )
        nobj = max(nobj, tnobj)
        tm_fit += ttm
        reslists[key] = reslist

    return reslists, nobj, tm_fit
    
def do_fits(sim,
            fit_conf,
            fitter,
            cat=None,
            seg=None,
            obs=None,
            show=False):

    fof_conf=fit_conf['fofs']
    weight_type=fit_conf['weight_type']

    if fof_conf['find_fofs']:
        logger.debug('extracting and finding fofs')
        if fof_conf.get('link_all',False):
            mbobs_list = sim.get_mbobs_list(
                cat=cat,
                seg=seg,
                obs=obs,
                weight_type=weight_type,
            )
            mbobs_list = [mbobs_list]
        else:
            mbobs_list = sim.get_fofs(
                fof_conf,
                cat=cat,
                seg=seg,
                obs=obs,
                weight_type=weight_type,
                show=show,
            )
    else:
        logger.debug('extracting')
        mbobs_list = sim.get_mbobs_list(
            cat=cat,
            seg=seg,
            obs=obs,
            weight_type=weight_type,
        )

    if len(mbobs_list)==0:
        reslist=None
        nobj=0
        tm_fit=0
        logger.debug("no objects detected")
    else:

        if fof_conf['find_fofs']:
            # mbobs_list is really a list of those
            reslist, nobj, tm_fit = run_fofs(fitter, mbobs_list)
        else:
            reslist, nobj, tm_fit = run_one_fof(fitter, mbobs_list)

        logger.debug("    processed %d objects" % nobj)

    return reslist, nobj, tm_fit

def run_fofs(fitter, fof_mbobs_lists):
    """
    run all fofs that were found
    """
    datalist=[]
    nobj=0
    tm=0.0

    nfofs=len(fof_mbobs_lists)
    logger.debug("processing: %d fofs" % nfofs)
    for i,mbobs_list in enumerate(fof_mbobs_lists):
        logger.debug("    fof: %d/%d has %d members" % (i+1,nfofs,len(mbobs_list)))
        reslist, t_nobj, t_tm = run_one_fof(fitter, mbobs_list)

        nobj += t_nobj
        tm += t_tm
        datalist += reslist

    return datalist, nobj, tm

def run_one_fof(fitter, mbobs_list):
    """
    running on a set of objects
    """

    nobj = len(mbobs_list)

    datalist=[]
    tm0=time.time()
    if nobj > 0:

        res=fitter.go(mbobs_list)
        if res is None:
            logger.debug("failed to fit")
        else:
            datalist.append(res)
    tm = time.time()-tm0

    return datalist, nobj, tm

def make_meta(ntrials,
              nsim,
              nfit,
              nobj_detected,
              nkept,
              elapsed_time,
              tm_sim,
              tm_fit):
    dt=[
        ('ntrials','i8'),
        ('nsim','i8'),
        ('nfit','i8'),
        ('nobj_detected','i8'),
        ('nkept','i8'),
        ('tm','f4'),
        ('tm_minutes','f4'),
        ('tm_per_trial','f4'),
        ('tm_sim','f4'),
        ('tm_fit','f4'),
        ('tm_per_sim','f4'),
        ('tm_per_fit','f4'),
        ('tm_per_obj_detected','f4'),
    ]
    meta=np.zeros(1, dtype=dt)
    meta['ntrials'] = ntrials
    meta['nsim'] = nsim
    meta['nfit'] = nfit
    meta['nobj_detected'] = nobj_detected
    meta['nkept'] = nkept
    meta['tm_sim'] = tm_sim
    meta['tm_fit'] = tm_fit
    meta['tm'] = elapsed_time
    meta['tm_minutes'] = elapsed_time/60
    meta['tm_per_sim'] = tm_sim/nsim
    meta['tm_per_trial'] = elapsed_time/ntrials

    if nfit > 0:
        tm_per_fit=tm_fit/nfit
    else:
        tm_per_fit=-9999

    if nobj_detected > 0:
        tm_per_obj_detected =tm_fit/nobj_detected
    else:
        tm_per_obj_detected=-9999

    meta['tm_per_fit'] = tm_per_fit
    meta['tm_per_obj_detected'] = tm_per_obj_detected

    return meta


def get_fitter(sim_conf, fit_conf, fitrng):
    """
    get the appropriate fitting class
    """
    if fit_conf['fitter']=='metacal':
        if fit_conf['fofs']['find_fofs']:
            mof_fitter = fitters.MOFFitter(fit_conf, sim_conf['nband'], fitrng)
        else:
            mof_fitter=None

        fitter=fitters.MetacalFitter(
            fit_conf,
            sim_conf['nband'],
            fitrng,
            mof_fitter=mof_fitter,
        )
    elif fit_conf['fitter']=='mof':
        fitter = fitters.MOFFitter(fit_conf, sim_conf['nband'], fitrng)

    elif fit_conf['fitter']=='max':
        fitter = fitters.MaxFitter(fit_conf, sim_conf['nband'], fitrng)
    else:
        raise ValueError("bad fitter: '%s'" % fit_conf['fitter'])

    return fitter


def profile_sim(seed,sim_conf,fit_conf,ntrials,output_file):
    """
    run the simulation using a profiler
    """
    import cProfile
    import pstats

    cProfile.runctx('go(seed,sim_conf,fit_conf,ntrials,output_file)',
                    globals(),locals(),
                    'profile_stats')
    
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()


def write_output(output_file, data, meta):
    """
    write an output file, making the directory if needed
    """
    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)
    with fitsio.FITS(output_file,'rw',clobber=True) as fits:
        fits.write(data, extname='model_fits')
        fits.write(meta, extname='meta_data')

def write_meta(output_file, datalist, meta, fit_conf):

    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)

    types=fit_conf['metacal']['metacal_pars']['types']
    with fitsio.FITS(output_file,'rw',clobber=True) as fits:
        for mtype in types:

            dlist=[]
            for d in datalist:
                if mtype in d:
                    # this is a list of results
                    dlist += d[mtype]

            if len(dlist) == 0:
                raise RuntimeError("no results found for type: %s" % mtype)

            data = eu.numpy_util.combine_arrlist(dlist)
            logger.info('    %s' % mtype)
            fits.write(data, extname=mtype)

        fits.write(meta, extname='meta_data')


