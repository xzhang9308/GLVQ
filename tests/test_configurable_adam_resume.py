from copy import deepcopy

import pytest
import torch

from src.configurable_adam import ConfigurableAdamW


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("betas,amsgrad", [((.9, .95), False), ((0., .95), False), ((.9, .95), True)])
def test_resume_preserves_moment_storage_and_next_updates(device, betas, amsgrad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required to verify CPU-offloaded optimizer resume")
    param = torch.nn.Parameter(torch.linspace(-1, 1, 19, device=device))
    kwargs = dict(lr=.003, betas=betas, amsgrad=amsgrad,
                  exp_avg_dtype=torch.float64, exp_avg_sq_dtype=torch.float64,
                  v_hat_max_dtype=torch.float64)
    if device == "cuda":
        kwargs.update(exp_avg_device=torch.device("cpu"), exp_avg_sq_device=torch.device("cpu"),
                      v_hat_max_device=torch.device("cpu"))
    original = ConfigurableAdamW([param], **kwargs)
    param.grad = torch.cos(param.detach())
    original.step()
    if device == "cuda":
        torch.cuda.synchronize()
    saved = deepcopy(original.state_dict())
    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = ConfigurableAdamW([resumed_param], **kwargs)
    resumed.load_state_dict(saved)
    for key, value in original.state[param].items():
        restored = resumed.state[resumed_param][key]
        if torch.is_tensor(value):
            assert restored.device == value.device
            assert restored.dtype == value.dtype
            assert restored.data_ptr() != saved["state"][0][key].data_ptr()
            if device == "cuda":
                assert restored.is_pinned()
            torch.testing.assert_close(restored, value, rtol=0, atol=0)
        else:
            assert restored == value
    for step in range(2):
        grad = torch.sin(torch.arange(19, device=device) + step)
        param.grad = grad.clone()
        resumed_param.grad = grad.clone()
        original.step()
        resumed.step()
        if device == "cuda":
            torch.cuda.synchronize()
        torch.testing.assert_close(resumed_param, param, rtol=0, atol=0)
    # Loading and subsequent training must not consume or mutate the checkpoint.
    assert saved["state"][0]["step"] == 1
