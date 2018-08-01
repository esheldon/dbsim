import os
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
       make_plots=False):

    rng = np.random.RandomState(seed)
    fitseed=rng.randint(0,2**30)
    fitrng = np.random.RandomState(fitseed)

    sim=simulation.Sim(sim_conf, rng)

    fclass=get_fitclass(run_conf)
    fitter=fclass(run_conf, sim_conf['nband'], fitrng)

    datalist=[]
    for i in range(ntrials):

        sim.make_obs()
        mbobs_list = sim.get_mbobs_list()
        if len(mbobs_list)==0:
            logger.debug("no objects detected")
        else:

            res=fitter.go(mbobs_list)
            if res is None:
                logger.debug("failed to fit")
            else:
                datalist.append(res)

    if len(datalist) == 0:
        logger.info("no results to write")
    else:
        data = eu.numpy_util.combine_arrlist(datalist)
        write_output(output_file, data)

def get_fitclass(conf):
    if conf['fitter']=='mof':
        fclass=fitters.MOFFitter
    elif conf['fitter']=='metacal':
        fclass=fitters.MetacalFitter
    else:
        raise ValueError("bad fitter: '%s'" % run_conf['fitter'])
    return fclass


def profile_sim(seed,sim_conf,run_conf,ntrials,output_file):
    import cProfile
    import pstats

    cProfile.runctx('go(seed,sim_conf,run_conf,ntrials,output_file)',
                    globals(),locals(),
                    'profile_stats')
    
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()


def write_output(output_file, data):

    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)
    fitsio.write(output_file, data, clobber=True)
