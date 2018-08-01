def read_yaml(fname):
    """
    read any yaml file
    """
    import yaml

    with open(fname) as fobj:
        data=yaml.load(fobj)

    return data


