from __future__ import annotations


def test_public_exports_importable() -> None:
    from delfos.reconstruct import ReconstructionService, TagFilter

    assert ReconstructionService.__name__ == "ReconstructionService"
    # TagFilter is a tuple type alias; just confirm it is importable and usable.
    _: TagFilter = ("language", "python")  # type: ignore[assignment]
