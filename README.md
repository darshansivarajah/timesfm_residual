# TimesFM Residual

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting. This GitHub repository supports our paper that builds on this model, utilizing a residual layer to make more accurate predictions. 

This open version is not an officially supported Google product.

**Model Version Used:** TimesFM 2.5

**Archived Model Versions:**


### Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/darshansivarajah/timesfm_residual.git
    cd timesfm_residual
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

    # Restart the terminal and navigate to the timesfm_residual folder
    
    # Create a virtual environment
    uv venv
    
    # Activate the environment
    source .venv/bin/activate # macOS
    .venv\Scripts\activate    # Windows
    
    # Install the package in editable mode with torch. If the lines do not run, try without the quotation marks
    uv pip install -e '.[torch]'
    # Or with flax
    uv pip install -e '.[flax]'
    # Or XReg is needed
    uv pip install -e '.[xreg]'
    ```

3. Update your pip. Install packages from requirements.txt, and a few additional packages/datasets. Create a token through Huggingface and enter it into the terminal when prompted.
```shell
    uv pip compile pyproject.toml -o requirements.txt

    python -m ensurepip --upgrade
    pip install u8darts

    # If errors are shown for any packages, use the following code line:
    uv pip install package # replacing package with the relevant package

    huggingface-cli login
    # Enter your token (input will not be visible): [enter token here]
    # Add token as git credential? (Y/n) [respond with y]
```

3. [Optional] Install your preferred `torch` / `jax` backend based on your OS and accelerators
(CPU, GPU, TPU or Apple Silicon).:

-   [Install PyTorch](https://pytorch.org/get-started/locally/).
-   [Install Jax](https://docs.jax.dev/en/latest/installation.html#installation)
    for Flax.

