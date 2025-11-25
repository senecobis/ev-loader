import torch

def custom_collate(batch):
    """
    Collate function for variable-length event data.
    Each item in the batch is a list (sequence) of data dicts.
    So we first want to collate over the outer batch dimension,
    and then manage the inhomogeneous fields.
    """
    seq_len = len(batch[0])  # assuming consistent seq_len across samples

    # Initialize lists to hold the data
    collated = [[] for _ in range(seq_len)]

    for b in batch:
        for i, item in enumerate(b):
            collated[i].append(item)

    # Now, for each time step, collate homogeneous fields and keep lists for inhomogeneous
    output = []
    for timestep_data in collated:
        timestep_output = {}
        keys = timestep_data[0].keys()

        for key in keys:
            items = [d[key] for d in timestep_data]

            if key in ["event_list", "d_event_list", "polarity_mask", "d_polarity_mask"]:
                # Keep these as a list (i.e. variable-length)
                timestep_output[key] = items
            elif isinstance(items[0], torch.Tensor) and items[0].ndim > 0:
                timestep_output[key] = torch.stack(items)
            else:
                timestep_output[key] = items

        output.append(timestep_output)

    return output

def noexception_collate(batch):
    """
    Collate function for variable-length event data.
    Each item in the batch is a list (sequence) of data dicts.
    So we first want to collate over the outer batch dimension,
    and then manage the inhomogeneous fields.
    """
    seq_len = len(batch[0])  # assuming consistent seq_len across samples

    # Initialize lists to hold the data
    collated = [[] for _ in range(seq_len)]

    for b in batch:
        for i, item in enumerate(b):
            collated[i].append(item)

    # Now, for each time step, collate homogeneous fields and keep lists for inhomogeneous
    output = []
    for timestep_data in collated:
        timestep_output = {}
        keys = timestep_data[0].keys()

        for key in keys:
            items = [d[key] for d in timestep_data]

            if isinstance(items[0], torch.Tensor) and items[0].ndim > 0:
                timestep_output[key] = torch.stack(items)
            else:
                timestep_output[key] = items

        output.append(timestep_output)

    return output