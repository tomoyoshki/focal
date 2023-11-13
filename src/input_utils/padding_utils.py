import math


def get_padded_size(img_size, window_size, patch_size, block_nums):
    r"""Calculate the padded image size based on the block number, window size, and image size
    Args:
        img_size [int, int]: Image size
        window_size [int, int]: Window size
        block_nums (int): Length of SwinTransformer blocks
    """
    # get the number of downsampling in the layer
    scale_factor = 2 ** (block_nums - 1)

    # find the minimum height and width that satisfies the downsampling
    scaled_height = window_size[0] * patch_size[0] * scale_factor
    scaled_width = window_size[1] * patch_size[1] * scale_factor
    scaled_size = [scaled_height, scaled_width]
    padded_img_size = [max(scaled_height, img_size[0]), max(scaled_width, img_size[1])]

    for i in range(2):
        if padded_img_size[i] % scaled_size[i] != 0:
            # find a size greater than img_size divisible by window_size and ([2 ** len(blocks))
            max_depth_len = math.ceil(padded_img_size[i] / scaled_size[i])

            # new_img_size = window_size * 2 ** x
            padded_img_size[i] = scaled_size[i] * max_depth_len
            
    return padded_img_size
