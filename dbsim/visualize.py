def make_rgb(r, g, b):
    import images
    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        nonlinear=0.1,
    )

    return rgb
