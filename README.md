# TimesFM Residual

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting. This GitHub repository supports our paper that builds on this model, utilizing a residual layer to make more accurate predictions. 

This open version is not an officially supported Google product.

**Model Version Used:** TimesFM 2.5

**Archived Model Versions:**


### Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/darshansivarajah/timesfm_residual.git
    cd timesfm
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```shell
    # If you don't have 'uv' yet, install it
    curl -Ls https://astral.sh/uv/install.sh | sh # macOS
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

    # alternatively, if you have Homebrew:
    brew install uv
    #or if you have Python
    pip install uv

    # Restart the terminal
    
    # Create a virtual environment
    uv venv
    
    # Activate the environment
    source .venv/bin/activate
    
    # Install the package in editable mode with torch
    uv pip install -e '.[torch]'
    # Or with flax
    uv pip install -e '.[flax]'
    # Or XReg is needed
    uv pip install -e '.[xreg]'
    ```

3. Update your pip. Install packages from requirements.txt, and a few additional packages/datasets. Create a token through Huggingface and enter it into the terminal when prompted.
```
uv pip compile pyproject.toml -o requirements.txt

python -m ensurepip --upgrade
pip install u8darts

huggingface-cli login
# Enter your token (input will not be visible): [enter token here]
# Add token as git credential? (Y/n) [respond with y]
```

3. [Optional] Install your preferred `torch` / `jax` backend based on your OS and accelerators
(CPU, GPU, TPU or Apple Silicon).:

-   [Install PyTorch](https://pytorch.org/get-started/locally/).
-   [Install Jax](https://docs.jax.dev/en/latest/installation.html#installation)
    for Flax.


### Code Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```
