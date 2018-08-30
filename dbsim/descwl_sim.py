"""
wrap the descwl simulations
"""
import os
import logging
import numpy as np
import fitsio
import galsim
import ngmix
import esutil as eu
from . import simulation
from . import pdfs
from . import visualize

logger = logging.getLogger(__name__)

class DESWLSim(simulation.Sim):
    def __init__(self,
                 config,
                 catalog_sampler,
                 position_sampler,
                 rng):

        self.update(config)

        self.catalog_sampler=catalog_sampler
        self.position_sampler=position_sampler

        self.rng=rng
        self.gs_rng = galsim.random.BaseDeviate(seed = self.rng.randint(0,2**30))

        self._set_view_area()
        self._set_dims()

    def make_obs(self, add_noise=True):
        """
        make a new MultiBandObsList, sampling from the catalog
        """
        cat=self.catalog_sampler.sample(area=self['area'])
        logger.debug('rendering: %d' % cat.size)

        # offsets in arcminutes
        pos_yx = self.position_sampler.sample(size=cat.size)

        mbobs = ngmix.MultiBandObsList()

        for band in self['bands']:
            dobs = self._make_band_dobs(cat, pos_yx, band, add_noise=add_noise)
            obs = self._convert_dobs_to_obs(dobs)
            #eu.stat.print_stats(obs.image.ravel())
            obslist=ngmix.ObsList()
            obslist.append(obs)

            mbobs.append(obslist)

        self.obs=mbobs
        self.cat=cat

        if 'background' in self:
            if self['background']['measure']:
                self._subtract_backgrounds()

    def get_psf_obs(self, band):
        """
        make a copy of the psf
        """
        return self.obs[band][0].psf.copy()

    def get_truth_catalog(self):
        """
        get a copy of the truth catalog
        """
        newcat=eu.numpy_util.add_fields(self.cat, [('image_id','i4')])
        return newcat

    def _set_psfs(self, mbobs_list, psf_obs=None):
        for mbobs in mbobs_list:
            for band,olist in enumerate(mbobs):
                for obs in olist:
                    if psf_obs is not None:
                        tpsf_obs=psf_obs.copy()
                    else:
                        tpsf_obs=self.get_psf_obs(band)
                    obs.set_psf(tpsf_obs)


    def _subtract_backgrounds(self):
        import sep
        c=self['background']['config']

        for obslist in self.obs:
            for obs in obslist:
                im=obs.image

                bkg = sep.Background(
                    im,
                    bw=c['back_size'],
                    bh=c['back_size'],
                    fw=c['filter_width'],
                    fh=c['filter_width'],
                )
                bkg_image = bkg.back()
                logger.debug("    bkg median: %g" % np.median(bkg_image))
                im -= bkg_image


    def _convert_dobs_to_obs(self, dobs):
        """
        convert a Survey object dobs to an ngmix Observation
        """
        j = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=dobs.pixel_scale,
        )

        im=dobs.image.array

        #print('mean sky level:',dobs.mean_sky_level )
        #print('sqrt(mean sky level):',np.sqrt(dobs.mean_sky_level ))
        noise = np.sqrt( dobs.mean_sky_level )
        weight = im*0 + 1.0/noise**2

        psf_im=dobs.psf_image.array.copy()
        psf_im = dobs.psf_model.drawImage(
            nx=32,
            ny=32,
            scale=dobs.pixel_scale,
        ).array
        psf_im *= 1.0/psf_im.sum()

        pnoise = psf_im.max()/400.0
        psf_s2n = np.sqrt( (psf_im**2).sum())/pnoise
        logger.debug('    psf s2n: %f' % psf_s2n)

        psf_im += self.rng.normal(scale=pnoise, size=psf_im.shape)

        psf_weight = psf_im*0 + 1.0/pnoise**2

        #import images
        #images.multiview(psf_im,title='psf')
        #stop

        psf_obs=ngmix.Observation(
            psf_im,
            weight=psf_weight,
            jacobian=j,
        )

        return ngmix.Observation(
            im,
            weight=weight,
            jacobian=j,
            psf=psf_obs,
        )

    def _make_band_dobs(self, catalog, pos_yx, band, add_noise=True):
        """
        make the image, which goes into 'dobs' and
        WeakLensingDeblending Survey instance
        """
        import descwl

        dobs, engine, builder = self._get_dobs_engine_builder(band)
        for i,entry in enumerate(catalog):

            # offsets in arcmin
            dy, dx = pos_yx[i]
            dy = dy*60.0
            dx = dx*60.0

            galaxy = builder.from_catalog(entry,dx,dy,dobs.filter_band)

            # this writes into the image contained in dobs
            # we throw away stamp,bounds
            try:
                stamps,bounds = engine.render_galaxy(
                    galaxy,
                    no_partials=True,
                    calculate_bias=False,
                )
            except descwl.render.SourceNotVisible as err:
                pass
                #print(str(err))

        if add_noise:
            noise = galsim.PoissonNoise(
                rng=self.gs_rng,
                sky_level = dobs.mean_sky_level,
            )
            dobs.image.addNoise(noise)
            #print('sky level:',dobs.mean_sky_level)
            #print('image median:',np.median(dobs.image.array))
            #dobs.image.array[:,:] -= dobs.mean_sky_level*dobs.pixel_scale**2

        return dobs

    def _get_dobs_engine_builder(self, band):
        import descwl

        dobs=self._get_dobs_by_band(band, dims=self['dims'])
        engine = descwl.render.Engine(
            survey=dobs,
            min_snr=0.05,
            #min_snr=0.00,
            truncate_radius=30,
            no_margin=False,
            verbose_render=False,
        )

        builder = descwl.model.GalaxyBuilder(
            survey=dobs,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )
        return dobs, engine, builder

    def _get_dobs_by_band(self, band, dims=None):
        import descwl

        pars=descwl.survey.Survey.get_defaults(
            survey_name=self['survey_name'],
            filter_band=band,
        )

        # survey_name and filter_band must be resent
        pars['survey_name'] = self['survey_name']
        pars['filter_band'] = band
        pars['image_width'] = dims[1]
        pars['image_height'] = dims[0]
        pars['cosmic_shear_g1'] = self['shear'][0]
        pars['cosmic_shear_g2'] = self['shear'][1]

        return descwl.survey.Survey(**pars)

    def _set_view_area(self):
        """
        this is generally smaller
        """

        self['area'] = self['positions']['width']**2

    def _set_dims(self):
        dobs=self._get_dobs_by_band(self['bands'][0], dims=[10,10])

        size_pixels=self['image_size_arcmin']*60.0/dobs.pixel_scale
        size_pixels=int(round(size_pixels))

        self['dims'] = [size_pixels]*2

class PositionSampler(dict):
    def __init__(self, config, rng):
        self.rng=rng
        self.update(config)
        self._set_pdf()

    def sample(self, size=None):
        return self.pdf.sample(size=size)

    def _set_pdf(self):
        assert self['type']=='uniform'

        half=self['width']/2.0
        self.pdf=pdfs.Flat2D(
            [-half, half],
            [-half, half],
            rng=self.rng,
        )

class CatalogSampler(dict):
    """
    class to sample from a catalog.  Currently
    the catalog path is expected to be in
    the CATSIM_DIR and called OneDegSq.fits
    """
    def __init__(self, config, rng):
        self.rng=rng
        self.update(config)
        self._load_catalog()
        self._set_density()

    def sample(self, nobj=None, area=None):
        """
        sample from the catalog

        either send the number of objects, or the area in arcmin^2 and the
        number of objects is taken to be density*area

        parameters
        ----------
        nobj: int, optional
            The user can request the number of objects
        area: float, optional
            The user can send the area in arcminutes^2 and
            the number will be generated from the catalog
            density
        """
        if nobj is None and area is None:
            raise ValueError('send area= or nobj=')

        if area is not None:
            nobj = int(round(area*self['density_arcmin2']))

        ind = self.rng.choice(
            self.cat_indices,
            size=nobj,
            replace=False,
        )
        # this makes a copy since the ind is not a slice
        return self.cat[ind]

    def _load_catalog(self):
        assert 'CATSIM_DIR' in os.environ

        dir=os.environ['CATSIM_DIR']
        fname=os.path.join(dir, 'OneDegSq.fits')
        logger.debug('reading: %s' % fname)
        self.cat=fitsio.read(fname)
        self.cat['pa_disk'] = self.rng.uniform(low=0.0, high=360.0, size=self.cat.size)
        self.cat['pa_bulge'] = self.cat['pa_disk']
        #import biggles
        #biggles.plot_hist(self.cat['pa_disk'],nbin=100)
        #stop
        self.cat_indices=np.arange(self.cat.size)

    def _set_density(self):
        nobj=self.cat.size

        area_deg2=1.0 # square degree
        area_arcmin2=area_deg2*60.0**2

        self['density_arcmin2']=nobj/area_arcmin2

        logger.debug('density of catalog: %.1f per square '
                     'arcminute' % self['density_arcmin2'])
