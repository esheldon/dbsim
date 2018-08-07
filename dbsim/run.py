import os
import time
import numpy as np
import logging
import fitsio
import esutil as eu
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

    fof_conf=fit_conf['fofs']

    sim=simulation.Sim(sim_conf, rng)

    fclass=get_fitclass(fit_conf)
    fitter=fclass(fit_conf, sim_conf['nband'], fitrng)

    nsim=0
    nfit=0
    nobj_detected=0
    tm0_main = time.time()
    tm_sim=0.0
    tm_fit=0.0

    weight_type=fit_conf['weight_type']

    datalist=[]
    for i in range(ntrials):

        logger.debug("trial: %d/%d" % (i+1,ntrials))

        tm0=time.time()
        sim.make_obs()
        if fof_conf['find_fofs']:
            mbobs_list = sim.get_fofs(fof_conf, weight_type=weight_type,
                                      show=show)
        else:
            mbobs_list = sim.get_mbobs_list(weight_type=weight_type)

        tm_sim += time.time()-tm0
        nsim += 1

        if show:
            sim.show()
            if 'q'==input('hit a key (q to quit): '):
                return

        if len(mbobs_list)==0:
            logger.debug("no objects detected")
        else:

            if fof_conf['find_fofs']:
                # mbobs_list is really a list of those
                reslist, nobj, tm = run_fofs(fitter, mbobs_list)
            else:
                reslist, nobj, tm = run_one_fof(fitter, mbobs_list)

            logger.debug("    processed %d objects" % nobj)
            nobj_detected += nobj
            datalist += reslist
            tm_fit += tm

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
        data = eu.numpy_util.combine_arrlist(datalist)
        write_output(output_file, data, meta)

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

def make_meta(ntrials, nsim, nfit, nobj_detected, nkept, elapsed_time, tm_sim, tm_fit):
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


def get_fitclass(conf):
    """
    get the appropriate fitting class
    """
    if conf['fitter']=='mof':
        fclass=fitters.MOFFitter
    elif conf['fitter']=='mof-metacal':
        fclass=fitters.MetacalFitter
    else:
        raise ValueError("bad fitter: '%s'" % conf['fitter'])
    return fclass


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
        fits.write(data, ext='model_fits')
        fits.write(meta, ext='meta_data')
