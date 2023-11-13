all_value_ranges = {
    "MOD": {
        "time": {
            "audio": 44778.1953125,
            "seismic": 71805.0,
        },
        "frequency": {
            "audio": 1023106.0,
            "seismic": 14450094.0,
        },
    },
}


def normalize_input(loc_inputs, args, level="time"):
    """Normalize the data between [-1, 1]"""
    normed_loc_inuts = {}

    if level == "feature":
        for loc in loc_inputs:
            max_abs = all_value_ranges[args.dataset][level][args.model][loc]
            normed_loc_inuts[loc] = loc_inputs[loc] / max_abs
    else:

        for loc in loc_inputs:
            normed_loc_inuts[loc] = {}
            for mod in loc_inputs[loc]:
                max_abs = all_value_ranges[args.dataset][level][mod]
                normed_loc_inuts[loc][mod] = loc_inputs[loc][mod] / max_abs

    return normed_loc_inuts
