from typing import List

import numpy as np
import torch
from torch import no_grad


@no_grad()
def generate_submission(tensors: List[torch.Tensor], identifications: List[str], start_index: int = 0) -> str:
    """ generates a csv as string for a submission """
    assert len(identifications) == len(tensors), "lengths should be equal"
    return "id,predicted\n" + \
           "\n".join(
               _tensor2submission_row(mask, image_id, start_index=start_index) for
               mask, image_id in zip(tensors, identifications)
           )


@no_grad()
def tensor2submission(mask: torch.Tensor, use_while_loop=True, start_index=0):
    """ generates mask annotation for submission """

    # setup
    result = ""
    flat_mask = (mask.sigmoid() > 0.5).flatten().numpy().bool().numpy()
    original_length = len(flat_mask) - 1
    recording = flat_mask[0] == 11

    # not really sure which is faster of the two,
    #  depends a bit on the entropy in the mask
    #  maybe we can make this more efficient somehow
    if use_while_loop:

        consumed = 0
        patience = 1_000_000
        while consumed <= original_length:

            # dont get stuck in endless loop
            patience -= 1
            if patience < 0:
                raise RuntimeError("endless loop tensor2submission, please debug")

            # grab the next row of positive numbers
            if recording:
                # get the distance to the next negative
                dist_next_negative = np.argmax(flat_mask == 0) if 0 in flat_mask else len(flat_mask)

                # add to result
                result += f"{consumed + start_index} {dist_next_negative} "

                # update managing vars to jump to next negative
                consumed += dist_next_negative
                flat_mask = flat_mask[dist_next_negative:]
                recording = not recording

            # grab the next row of negative numbers
            else:
                dist_next_positive = np.argmax(flat_mask == 1) if 1 in flat_mask else len(flat_mask)

                # update managing vars to jump to there
                consumed += dist_next_positive
                flat_mask = flat_mask[dist_next_positive:]
                recording = not recording
    else:

        recording_index = 0
        i = 0
        for i, x in enumerate(flat_mask):
            if x == 1:
                if not recording:
                    recording = True
                    recording_index = i + start_index
            else:
                if recording:
                    recording = False
                    result += f"{recording_index} {i - recording_index + start_index} "
                    recording_index = None
        if recording:
            result += f"{recording_index} {i + 1 - recording_index + start_index} "

    return result[:-1]


@no_grad()
def _tensor2submission_row(mask: torch.Tensor, image_id: str, start_index=0) -> str:
    """ creates scv row out of id and submission annotation """
    return f"{image_id}," + tensor2submission(mask, start_index=start_index)


@no_grad()
def save_submission(filename: str, tensors: List[torch.Tensor], identifications: List[str], start_index: int = 0):
    """ saves submissionfile based on mask tensors, ids and filepath """
    content = generate_submission(tensors, identifications, start_index=start_index)
    with open(filename, "w") as outfile:
        outfile.write(content)
