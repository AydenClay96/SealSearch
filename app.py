"""Streamlit interface for finding visually similar seal identities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from network import IMAGE_EXTENSIONS, Prediction, predict_seal


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "output"
CHECKPOINT_PATH = PROJECT_DIR / "seal_model.pth"


@st.cache_data(show_spinner=False)
def _predict_upload(image_bytes: bytes) -> list[Prediction]:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return predict_seal(image, CHECKPOINT_PATH, top_k=10)


def _first_reference_image(seal_id: str) -> Path | None:
    class_dir = DATA_DIR / "train" / seal_id
    if not class_dir.is_dir():
        return None
    return next(
        (
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        None,
    )


def main() -> None:
    st.set_page_config(page_title="SealSearch", page_icon="🦭", layout="wide")
    st.title("SealSearch")
    st.write(
        "Upload a seal photo to see the 10 closest seal identities in the dataset."
    )

    uploaded_file = st.file_uploader(
        "Choose a seal image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )
    if uploaded_file is None:
        return
    if not CHECKPOINT_PATH.is_file():
        st.error(
            f"Model index not found at {CHECKPOINT_PATH}. "
            "Build it first with `uv run python network.py train --data output`."
        )
        return

    image_bytes = uploaded_file.getvalue()
    uploaded_image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if uploaded_image is None:
        st.error("The uploaded file could not be decoded as an image.")
        return
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

    try:
        with st.spinner("Comparing the image with the seal index..."):
            predictions = _predict_upload(image_bytes)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not identify this image: {error}")
        return

    if not predictions:
        st.error("The model returned no seal matches.")
        return

    st.subheader("Closest match")
    uploaded_column, match_column = st.columns(2)
    with uploaded_column:
        st.image(
            uploaded_image,
            caption=f"Uploaded image: {uploaded_file.name}",
            use_container_width=True,
        )

    best_match = predictions[0]
    reference_image = _first_reference_image(best_match.seal_id)
    with match_column:
        if reference_image is None:
            st.warning(
                f"No training image was found for {best_match.seal_id} "
                f"under {DATA_DIR / 'train'}."
            )
        else:
            st.image(
                str(reference_image),
                caption=f"Training example for {best_match.seal_id}",
                use_container_width=True,
            )
    st.markdown(
        f"**Top match: {best_match.seal_id}** "
        f"(cosine similarity {best_match.similarity:.3f})"
    )

    st.subheader("Top 10 seal matches")
    st.caption("Scores are cosine similarities, not probabilities.")
    st.table(
        [
            {
                "Rank": rank,
                "Seal ID": prediction.seal_id,
                "Cosine similarity": f"{prediction.similarity:.3f}",
            }
            for rank, prediction in enumerate(predictions, start=1)
        ]
    )


if __name__ == "__main__":
    main()
