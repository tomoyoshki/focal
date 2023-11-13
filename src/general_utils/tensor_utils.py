import torch


def bisc_to_bcis(loc_mod_features):
    """Convert the shape of loc_mod_features from [b, i, s ,c] to [b, c, i, s].
       Transformer + feature level only.
    Args:
        loc_mod_features (_type_): _description_
    """
    output_loc_mod_features = dict()

    for loc in loc_mod_features:
        output_loc_mod_features[loc] = torch.permute(loc_mod_features[loc], (0, 3, 1, 2))

    return output_loc_mod_features


def bcis_to_bisc(loc_mod_features):
    """Convert the shape of loc_mod_features from [b, c, i ,s] to [b, i, s, c].
       Transformer + feature level only.
    Args:
        loc_mod_features (_type_): _description_
    """
    output_loc_mod_features = dict()

    for loc in loc_mod_features:
        output_loc_mod_features[loc] = torch.permute(loc_mod_features[loc], (0, 2, 3, 1))

    return output_loc_mod_features


def miss_ids_to_masks_input(miss_ids, num_mods, target_shapes, device):
    """Generate the miss_masks with the same shape as the target_shape.
    Note: In masks, 1 means available, 0 means missing.

    Args:
        miss_ids (_type_): [[miss_mod_ids] for each sample]
        target_shape (_type_): [b, c, i, spectrum]
        device (_type_): _description_
    Return:
        masks: dict of {mod_id: [b, c, i, s]}
    """
    masks = dict()
    for mod_id in range(num_mods):
        masks[mod_id] = torch.ones(target_shapes[mod_id]).to(device)

    for sample_id, sample_miss_ids in enumerate(miss_ids):
        for mod_id in sample_miss_ids:
            masks[mod_id][sample_id] = 0

    return masks


def miss_ids_to_masks_feature(miss_ids, target_shape, device):
    """Generate the miss_masks with the same shape as the target_shape.
       Note: In masks, 1 means available, 0 means missing.

    Args:
        miss_ids (_type_): [[miss_ids] for each sample]
        sensors (_type_): _description_
        target_shape (_type_): [b, c, i, s]
        device (_type_): _description_
    Return:
        masks: [b, c, i, s]
    """
    b, c, i, s = target_shape
    masks = torch.ones([b, s, c, i]).to(device)

    for sample_id, sample_miss_ids in enumerate(miss_ids):
        masks[sample_id][sample_miss_ids] = 0

    # [b, s, c, i] --> [b, c, i, s]
    masks = torch.permute(masks, (0, 2, 3, 1))

    return masks


def masks_to_miss_ids_feature(miss_masks):
    """Generate the miss sensors ids from the miss masks.
    NOTE: The batch dimension should be 1.

    Args:
        miss_masks (_type_): [b, c, i, s]
    """
    # We only take the first sample
    if miss_masks.ndim == 4:
        sample_mask = miss_masks[0]
    else:
        sample_mask = miss_masks

    sample_mask = torch.mean(torch.permute(sample_mask, [2, 0, 1]), dim=[1, 2])
    avl_ids = torch.nonzero(sample_mask).flatten()
    miss_ids = torch.nonzero(sample_mask == 0).flatten()

    return avl_ids, miss_ids


def masks_to_miss_ids_input(miss_masks):
    """Generate the miss sensors ids from the miss masks.
    NOTE: The batch dimension should be 1.

    Args:
        miss_masks (_type_): {mod_id: [b, c, i, spectrum]}
    """
    avl_ids, miss_ids = [], []
    for mod_id in miss_masks:
        if miss_masks[mod_id].min() == 1:
            avl_ids.append(mod_id)
        else:
            miss_ids.append(mod_id)

    return avl_ids, miss_ids


def select_with_mask(input_feature, miss_masks):
    """Only preserve the elements with positive mask values.

    Args:
        x (_type_): []b, c, i, s]
        miss_masks (_type_): [b, c, i, s], some modalities can be missing, but each sample have same #missing mods.
    """
    sample_features = torch.split(input_feature, 1)
    out_sample_features = []
    for sample_id, sample_feature in enumerate(sample_features):
        avl_ids, miss_ids = masks_to_miss_ids_feature(miss_masks[sample_id])
        out_sample_feature = torch.index_select(sample_feature, dim=3, index=avl_ids)
        out_sample_features.append(out_sample_feature)

    output_feature = torch.cat(out_sample_features, dim=0)

    return output_feature


def select_with_rescale_factors_bcis(input_feature, rescale_factors):
    """Only preserve the modalities with positive rescale factors.
    Args:
        x: [b, c, i, s] feature only
        rescale_factors: [b, s]
    """
    b, c, i, s = input_feature.shape
    expand_recale_factors = rescale_factors.reshape([b, 1, 1, s]).tile([1, c, i, 1])
    rescale_mask = expand_recale_factors > 0
    flatten_output_feature = torch.masked_select(input_feature, rescale_mask)
    output_feature = flatten_output_feature.reshape([b, c, i, -1])

    return output_feature


def select_with_rescale_factors_bisc(input_feature, rescale_factors):
    """Only preserve the modalities with positive rescale factors.
    Args:
        x: [b, i, s, c] feature only
        rescale_factors: [b, s]
    """
    b, i, s, c = input_feature.shape
    expand_recale_factors = rescale_factors.reshape([b, 1, s, 1]).tile([1, i, 1, c])
    rescale_mask = expand_recale_factors > 0
    flatten_output_feature = torch.masked_select(input_feature, rescale_mask)
    output_feature = flatten_output_feature.reshape([b, i, -1, c])

    return output_feature


def replace_avail_input(recon_input, org_input, miss_masks, args):
    """Replace with the original representations at the available samples."""
    if args.miss_detector != "FakeDetector":
        recon_mask = torch.ones_like(miss_masks) - miss_masks
        merged_input = org_input * miss_masks + recon_input * recon_mask
    else:
        merged_input = recon_input

    return merged_input


def zero_noisy_input(loc_mod_features, dt_loc_miss_masks):
    """Convert the detected missing modalities to 0."""
    for loc in loc_mod_features:
        loc_mod_features[loc] = loc_mod_features[loc] * dt_loc_miss_masks[loc]

    return loc_mod_features


def loc_rescale_factors_from_miss_masks(loc_miss_masks, args):
    """Generate the rescale factors according to the miss masks.
    Only used with the separte infernece mode under GT miss masks.
    loc_miss_masks: [b, c, i, sensor] at feature level; {mod_id: [b, c, i, spectrum]} at input level.
    """
    if args.noise_position == "feature":
        rescale_factors = torch.mean(loc_miss_masks, dim=[1, 2])
    else:
        rescale_factors = []
        for mod_id in loc_miss_masks:
            rescale_factors.append(loc_miss_masks[mod_id].mean(dim=[1, 2, 3]))
        rescale_factors = torch.stack(rescale_factors, dim=1)

    return rescale_factors


def extract_non_diagonal_matrix(input):
    """
    Extract the non-diagonal elements from the input matrix at the last two dimensions.
    input shape: [b, n, n]
    """
    flatten_input = input.reshape([-1, input.shape[-2], input.shape[-1]])
    b, n, _ = flatten_input.shape

    non_diagonal_input = flatten_input.flatten(start_dim=1)[:, 1:]
    non_diagonal_input = non_diagonal_input.view(b, n - 1, n + 1)[:, :, :-1]
    non_diagonal_input = non_diagonal_input.reshape([b, n, n - 1])

    return non_diagonal_input
