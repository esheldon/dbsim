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
       run_conf,
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

    fclass=get_fitclass(run_conf)
    fitter=fclass(run_conf, sim_conf['nband'], fitrng)

    nsim=0
    nfit=0
    tm0 = time.time()
    tm_sim=0.0
    tm_fit=0.0

    datalist=[]
    for i in range(ntrials):

        tm0=time.time()
        sim.make_obs()
        mbobs_list = sim.get_mbobs_list()
        tm_sim += time.time()-tm0
        nsim += 1

        if show:
            sim.show()
            if 'q'==input('hit a key (q to quit): '):
                return

        if len(mbobs_list)==0:
            logger.debug("no objects detected")
        else:


            tm0=time.time()
            res=fitter.go(mbobs_list)
            tm_fit += time.time()-tm0
            nfit += 1

            if res is None:
                logger.debug("failed to fit")
            else:
                datalist.append(res)

    elapsed_time=time.time()-tm0
    nkept = len(datalist)

    print("kept: %d/%d %.2f" % (nkept,ntrials,float(nkept)/ntrials))
    print("time minutes:",elapsed_time/60.0)
    print("time per sim:",tm_sim/nsim)
    if nfit > 0:
        print("time per fit:",tm_fit/nfit)

    if nkept == 0:
        logger.info("no results to write")
    else:
        data = eu.numpy_util.combine_arrlist(datalist)
        write_output(output_file, data)

def get_fitclass(conf):
    """
    get the appropriate fitting class
    """
    if conf['fitter']=='mof':
        fclass=fitters.MOFFitter
    elif conf['fitter']=='metacal':
        fclass=fitters.MetacalFitter
    else:
        raise ValueError("bad fitter: '%s'" % run_conf['fitter'])
    return fclass


def profile_sim(seed,sim_conf,run_conf,ntrials,output_file):
    """
    run the simulation using a profiler
    """
    import cProfile
    import pstats

    cProfile.runctx('go(seed,sim_conf,run_conf,ntrials,output_file)',
                    globals(),locals(),
                    'profile_stats')
    
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()


def write_output(output_file, data):
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
    fitsio.write(output_file, data, clobber=True)
