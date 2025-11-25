import numpy as np
import torch


def norm_p(p):
    p = p.astype(np.int8)
    p[p < 1] = -1
    return p

def norm_t(t):
    t = (t - t[0]) / (t[-1] - t[0])
    return t

def create_polarity_mask(ps):
    """
    Creates a two channel tensor that acts as a mask for the input event list.
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [N x 2] polarity list event representation
    """
    ps = torch.from_numpy(ps)
    event_list_pol_mask = torch.stack([ps, ps])
    event_list_pol_mask[0, :][event_list_pol_mask[0, :] < 0] = 0
    event_list_pol_mask[0, :][event_list_pol_mask[0, :] > 0] = 1
    event_list_pol_mask[1, :][event_list_pol_mask[1, :] < 0] = -1
    event_list_pol_mask[1, :][event_list_pol_mask[1, :] > 0] = 0
    event_list_pol_mask[1, :] *= -1
    return event_list_pol_mask

def split_events(x_rect, y_rect, p, t, max_num_grad_events, max_num_detach_events):
    p = norm_p(p)
    t = norm_t(t)

    # Split the event list into two lists, one of them 
    # (with max. length) to be used for backprop the other just for loss computation
    ev_list_pol_mask = create_polarity_mask(p)
    raw_ev = torch.from_numpy(np.stack([t, y_rect, x_rect, p], axis=0))

    # Split events for backprop and events just for calculation (detached)
    # TODO load always the same number of events for batching purpose
    ev_list, pol_mask, d_ev_list, d_pol_mask = _split_event_list(
        raw_ev, ev_list_pol_mask, max_num_grad_events, max_num_detach_events)

    # Flip to have the same convention of the Iterative Warping loss
    ev_list = ev_list.permute(1, 0)
    pol_mask = pol_mask.permute(1, 0)
    d_ev_list = d_ev_list.permute(1, 0)
    d_pol_mask = d_pol_mask.permute(1, 0)
    return ev_list, pol_mask, d_ev_list, d_pol_mask

def sample_with_equal_prob(size, num_samples, replacement=False):
    if replacement:
        sampled = torch.randint(low=0, high=size, size=(num_samples,), dtype=torch.long)
    else:
        sampled = torch.randperm(size)[:num_samples]
    return sampled

def _split_event_list(event_list, event_list_pol_mask, max_num_grad_events, max_num_detach_events):
    """
    Splits the event list into two lists, one of them (with max. length) to be used for backprop.
    This helps reducing (VRAM) memory consumption.
    :param event_list: [4 x N] list event representation
    :param event_list_pol_mask: [2 x N] polarity list event representation
    :param max_num_grad_events: maximum number of events to be used for backprop
    :return event_list: [4 x N] list event representation to be used for backprop
    :return event_list_pol_mask: [2 x N] polarity list event representation to be used for backprop
    :return d_event_list: [4 x N] list event representation
    :return d_event_list_pol_mask: [2 x N] polarity list event representation
    """

    # TODO load always the same number of events for batching purpose
    num_of_events = event_list.shape[1]

    if max_num_grad_events is None:
        return event_list, event_list_pol_mask, torch.zeros((4, 0)), torch.zeros((2, 0))

    if num_of_events > max_num_grad_events:
        indices = sample_with_equal_prob(num_of_events, max_num_grad_events, replacement=False)
    else:
        # print("Not enough events for GRADIENT, sampling with replacement")
        indices = sample_with_equal_prob(num_of_events, max_num_grad_events, replacement=True)

    event_list_ = event_list[:, indices]
    event_list_pol_mask_ = event_list_pol_mask[:, indices]
    
    unsampled_indices = torch.ones(num_of_events, dtype=torch.bool)
    unsampled_indices[indices] = False
    unsampled_event_list = event_list[:, unsampled_indices]
    unsampled_event_list_pol_mask = event_list_pol_mask[:, unsampled_indices]
    num_of_unsampled_events = unsampled_event_list.shape[1]

    if num_of_unsampled_events > max_num_detach_events:
        d_indices = sample_with_equal_prob(num_of_unsampled_events, max_num_detach_events, replacement=False)
    elif num_of_unsampled_events == 0:
        # sample event from the original list multiple times
        # if there are no remaining events
        d_indices = sample_with_equal_prob(num_of_events, max_num_detach_events, replacement=True)
        unsampled_event_list = event_list
        unsampled_event_list_pol_mask = event_list_pol_mask
    else:
        # print("Not enough events for LOSS COMPUTE, sampling with replacement")
        d_indices = sample_with_equal_prob(num_of_unsampled_events, max_num_detach_events, replacement=True)
    
    d_event_list = unsampled_event_list[:, d_indices]
    d_event_list_pol_mask = unsampled_event_list_pol_mask[:, d_indices]
    
    return event_list_, event_list_pol_mask_, d_event_list, d_event_list_pol_mask