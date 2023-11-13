def calc_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    """Evaluate the contrastive loss."""
    if args.learn_framework == "FOCAL":
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(
            aug_freq_loc_inputs_1, aug_freq_loc_inputs_2, proj_head=True
        )
        loss = loss_func(feature1, feature2)
    else:
        raise Exception(f"Invalid framework provided: {args.learn_framework}")

    return loss

def calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    """Choose the corrent loss function according to the train mode."""
    if args.train_mode == "contrastive":
        loss = calc_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs)
    else:
        raise Exception(f"Invalid train mode: {args.train_mode}")

    return loss
