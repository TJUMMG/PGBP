import torch

def find_most_relevant_frame(probabilities, mask, window_size):
    """
    This function finds the most relevant frame in a batch of videos based on the probabilities of each frame
    being relevant to the text. It uses a sliding window approach to find a continuous sequence of frames
    with the highest average probability. The mask ensures that only valid values are considered.

    :param probabilities: Batched tensor of probabilities (shape: [B, L]).
    :param mask: Batched tensor of masks (shape: [B, L]) where 1 indicates a valid value and 0 indicates invalid.
    :param window_size: Size of the sliding window.
    :return: The index of the frame with the highest probability for each batch.
    """
    batch_size, L = probabilities.shape

    # Initialize arrays to store results
    indices_of_max_frames = torch.zeros(batch_size, dtype=int).cuda()
    visual_len = torch.sum(mask,dim=1).long()
    for batch_index in range(batch_size):
        # Slide the window across the valid probabilities
        max_avg_probability = 0
        index_of_max_frame = 0
        probability = probabilities[batch_index]
        if visual_len[batch_index] < window_size:
            index_of_max_frame = torch.max(probability[0:visual_len[batch_index]],dim = 0)[1]
        else:
            for start_index in range(visual_len[batch_index] - window_size + 1):
                # Compute the average probability for the current window
                window_avg = torch.mean(probability[start_index:start_index + window_size])

                # If the current window's average probability is greater than the max found so far, update max
                if window_avg > max_avg_probability:
                    max_avg_probability = window_avg
                    index_of_max_frame = torch.max(probability[start_index:start_index + window_size],dim = 0)[1]
                    index_of_max_frame = index_of_max_frame + start_index
        indices_of_max_frames[batch_index] = index_of_max_frame
        if (indices_of_max_frames >= visual_len).any():
             print("indices_of_max_frames out of boundary")
    return indices_of_max_frames

# Example usage:

# torch.manual_seed(42)  # For reproducibility
# B = 3  # Batch size
# L = 100  # Length of each video in frames
# probabilities_batched = torch.rand(B, L)  # Random probabilities
# mask_batched = torch.randint(0, 2, size=(B, L))  # Random binary mask
# # Define a window size, e.g., corresponding to 1 second of video at 30 fps
# window_size = 30

# # Find the index of the most relevant frame for each batch
# index_of_max_frames_batched = find_most_relevant_frame_batched(probabilities_batched, mask_batched, window_size)

# index_of_max_frames_batched
def get_neg_sample(pos_ind,mask,pred):
    B,L = mask.shape
    mask1 = mask.clone()
    for i in range(B):
        mask1[i, pos_ind[i]:] = 0.0
    mask2 = mask-mask1
    neg1_value,neg1 = torch.min(pred.masked_fill(~mask1.bool(), float('1.0')), dim=1)
    neg2_value,neg2 = torch.min(pred.masked_fill(~mask2.bool(), float('1.0')), dim=1)   
    condition1 = (neg1_value == 1.0)
    neg1 = torch.where(condition1, neg2, neg1)
    condition2 = (neg2_value == 1.0)
    neg2 = torch.where(condition2, neg1, neg2)
    return neg1,neg2