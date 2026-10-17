# Seal identification neural network

The seal identifier in `network.py` uses a frozen ImageNet-pretrained ResNet-50
to encode coat-pattern details, then matches each query to its most similar
training image by cosine similarity. This nearest-reference approach performed
better than fine-tuning a classifier on the current small dataset. Training
builds and saves an embedding index from class-named folders in `output/train`,
reports validation metrics, and evaluates top-1/top-3 accuracy on
`output/test`. Images are cached at 448px and exact duplicate files are
excluded from later splits to prevent validation/test leakage. Install the
project dependencies with `uv sync`, then train from the repository root:

```powershell
uv run python network.py train --data output
```

The first run downloads the pretrained ResNet-50 weights if they are not
already cached. The model weights and reference embeddings are saved as
`seal_model.pth`. Identify an image with:

```powershell
uv run python network.py predict path\to\seal.jpg --top-k 3
```

Prediction scores are cosine similarities, not calibrated probabilities.

Launch the image-upload app from the repository root:

```powershell
uv run streamlit run app.py
```

Upload an image to view its top 10 seal matches. The app displays the uploaded
image beside a training-set example from the top-ranked seal identity.

The class folder names (for example, `Seal_SealID_3`) are returned as the
predictions. The classifier only recognizes identities present in the training
split; it does not detect seals in arbitrary scenes.

On Windows, `uv sync` selects the CUDA 13.2 PyTorch wheel so a compatible
NVIDIA GPU is used automatically. The first sync downloads a large (about 2 GB)
GPU-enabled wheel; subsequent syncs use the local cache. You do not need to
install the CUDA toolkit; the NVIDIA display driver is required. On other
platforms, PyTorch is installed from PyPI.