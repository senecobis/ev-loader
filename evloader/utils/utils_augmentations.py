import torch
import numpy as np
import albumentations as A

class EventListAugmentor:
    def __init__(self, width, height, hflip_prob=0.0, vflip_prob=0.0, polarity_flip_prob=0.0):
        self.width = width
        self.height = height
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.polarity_flip_prob = polarity_flip_prob

        self.last_flip_state = {"hflip": False, "vflip": False}

    def augment_mask(self, mask):
        mask = mask.permute(1,2,0).numpy()

        transforms = []
        if self.last_flip_state["hflip"]:
            transforms.append(A.HorizontalFlip(p=1.0))
        if self.last_flip_state["vflip"]:
            transforms.append(A.VerticalFlip(p=1.0))
        composed = A.Compose(
            transforms, additional_targets={"mask": "mask"})
        augmented_flow = composed(image=mask, mask=mask)

        mask = augmented_flow["mask"]
        mask = torch.from_numpy(mask).permute(2,0,1)
        return mask


    def augment_dense_data(self, flow1, flow2, mask):
        flow1 = flow1.permute(1,2,0).numpy()
        flow2 = flow2.permute(1,2,0).numpy()
        mask = mask.permute(1,2,0).numpy()

        transforms = []
        if self.last_flip_state["hflip"]:
            transforms.append(A.HorizontalFlip(p=1.0))
        if self.last_flip_state["vflip"]:
            transforms.append(A.VerticalFlip(p=1.0))
        composed = A.Compose(
            transforms, additional_targets={"flow1": "mask", "flow2": "mask", "mask": "mask"})
        augmented_flow = composed(image=flow1, flow1=flow1, flow2=flow2, mask=mask)

        flow1 = augmented_flow["flow1"]
        flow2 = augmented_flow["flow2"]
        mask = augmented_flow["mask"]

        flow1 = torch.from_numpy(flow1).permute(2,0,1)
        flow2 = torch.from_numpy(flow2).permute(2,0,1)
        mask = torch.from_numpy(mask).permute(2,0,1)
        return flow1, flow2, mask

    def __call__(self, x, y, p):
        """
        Args:
            events: np.ndarray of shape (N, 3) -> [x, y, polarity]
        Returns:
            Augmented events with same shape
        """
        x = x.copy()
        y = y.copy()
        p = p.copy()

        self.last_flip_state["hflip"] = np.random.rand() < self.hflip_prob
        self.last_flip_state["vflip"] = np.random.rand() < self.vflip_prob
        polarity_flip = np.random.rand() < self.polarity_flip_prob

        if self.last_flip_state["hflip"]:
            x = self.width - 1 - x  # Flip X

        if self.last_flip_state["vflip"]:
            y = self.height - 1 - y  # Flip Y

        # Polarity flip is polarity is 0 or 1
        if polarity_flip:
            p = 1-p  # Flip polarity

        return x, y, p
