def make_rgb(r, g, b):
    import images

    maxval=max( r.max(), g.max(), b.max() )
    scales = [1.0/maxval]*3
    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=0.1,
    )

    return rgb
