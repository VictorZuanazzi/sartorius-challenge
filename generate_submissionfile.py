import pathlib

import dask
import torch



def _is_neuron(x, threshold=1e-6):
    return x >= (1 - threshold)


def generate_submission(tensors, identifications, start_index=0):
    assert len(identifications) == len(tensors), "lengths should be equal"
    return "id,predicted\n" + dask.delayed(str.join)(
        "",
        (
            tensor2submissionrow(y_hat, image_id, start_index=start_index) for
            y_hat, image_id in zip(tensors, identifications)
        )
    ).compute()


@dask.delayed
def tensor2submissionrow(y_hat: torch.Tensor, image_id: str, start_index=0) -> str:
    flat_y = y_hat.flatten().cpu().detach().numpy()

    result = f"{image_id},"

    recording = False
    recording_index = None

    for i, x in enumerate(flat_y):
        if _is_neuron(x):
            if not recording:
                recording = True
                recording_index = i + start_index
        else:
            if recording:
                recording = False
                result += f"{recording_index} {i - recording_index} "
                recording_index = None

    return result + "\n"


    # print(flat_y)
    # print(result)


# print(generate_submission([(torch.randn((16, 9)) > 0).float(), (torch.randn((16, 9)) > 0).float()],
#                           ["banana_picture", "uhsauhsa"]))
#
# tensor2submissionrow((torch.randn((16, 9)) > 0).float(), "banana_picture")
