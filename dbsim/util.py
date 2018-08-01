import sys
import logging

logger = logging.getLogger(__name__)

class TryAgainError(Exception):
    """
    signal to skip this image(s) and try a new one
    """
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)

def setup_logging(level):
    if level=='info':
        l=logging.INFO
    elif level=='debug':
        l=logging.DEBUG
    elif level=='warning':
        l=logging.WARNING
    elif level=='error':
        l=logging.ERROR
    else:
        l=logging.CRITICAL

    logging.basicConfig(stream=sys.stdout, level=l)

def log_pars(pars, fmt='%8.3g',front=None):
    """
    print the parameters with a uniform width
    """

    s = []
    if front is not None:
        s.append(front)
    if pars is not None:
        fmt = ' '.join( [fmt+' ']*len(pars) )
        s.append( fmt % tuple(pars) )
    s = ' '.join(s)

    logger.debug(s)

class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front=='':
            front=None
        if back=='' or back=='noshear':
            back=None

        self.front=front
        self.back=back

        if self.front is None and self.back is None:
            self.nomod=True
        else:
            self.nomod=False



    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)
        
        return n

def convert_run_to_seed(run):
    """
    convert the input config file name to an integer for use
    as a seed
    """
    import hashlib

    h = hashlib.sha256(run.encode('utf-8')).hexdigest()
    seed = int(h, base=16) % 2**30 

    logger.info("got seed %d from run %s" % (seed,run))

    return seed

