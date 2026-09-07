"""Every `geofix_*` route the TRAINER resolves must be a parameter of the VALIDATOR.

`da3_validation.validate_da3_multiview` carries a comment saying this file exists
and that adding a training-time route without adding it to that signature "now
fails a test instead of producing a plausible curve". **It did not exist.** The
comment was written alongside the signature and the test was never committed, so
for every route added since, the promised guard was a sentence.

It cost exactly what the comment predicted, on 2026-09-07. `mask_in_channels` --
the widening route -- was added to the trainer and not to the validator, whose
`geofix_needs_mask` therefore stayed false; the mask was never bound, and the
first validation pass died inside `_append_geofix_mask`. That failure was LOUD
only because the widened embedder refuses to run without its planes. The camera
route has no such backstop: the same omission there produces a validation curve
scored with the mask silently off, which is a plausible number and wrong.

The test is reflection over both sides rather than a hand-kept list, because a
hand-kept list is the thing that drifted.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "train_multiview_da3.py"

#: Settings the trainer resolves that are deliberately NOT validator parameters,
#: each with the reason. Anything else must appear in the signature.
EXEMPT = {
    # Slot 1's ablation flag, not a route: it only forces `cond_artifact` off, and
    # `cond_artifact` itself IS a validator parameter.
    "no_mask",
    "no_cond_artifact",
}


def _trainer_routes() -> set[str]:
    """Every `geofix_cfg.get("<name>", ...)` the trainer reads."""
    src = TRAINER.read_text()
    return {m.group(1) for m in re.finditer(r'geofix_cfg\.get\(\s*["\'](\w+)["\']', src)}


def _validator_params() -> set[str]:
    """Parameter names of `validate_da3_multiview`, read with AST.

    Deliberately NOT `inspect.signature` on an import: importing that module
    pulls torch, the RAE and LPIPS, which makes a guard against a five-character
    omission cost a GPU node and two minutes. The signature is a syntactic fact
    and AST reads it in milliseconds anywhere.
    """
    tree = ast.parse((ROOT / "src" / "utils" / "da3_validation.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_da3_multiview":
            a = node.args
            return {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    raise AssertionError("validate_da3_multiview not found")


def test_every_trainer_route_is_a_validator_parameter():
    params = _validator_params()
    missing = sorted(
        name for name in _trainer_routes()
        if name not in EXEMPT and f"geofix_{name}" not in params
    )
    assert not missing, (
        f"the trainer resolves geofix.{{{', '.join(missing)}}} but "
        "validate_da3_multiview has no parameter for them, so validation runs "
        "with those routes OFF while training runs with them ON. Add "
        f"{', '.join('geofix_' + m for m in missing)} to the signature, include "
        "them in `geofix_needs_*`, and pass them from the trainer's call."
    )


def test_the_routes_that_consume_a_mask_all_reach_needs_mask():
    """`geofix_needs_mask` must name every mask-consuming parameter it has.

    A route in the signature but absent from `geofix_needs_mask` is the SILENT
    half of the same bug: the parameter exists, the caller passes it, and the
    mask is still never bound.
    """
    val = (ROOT / "src" / "utils" / "da3_validation.py").read_text()
    block = re.search(r"geofix_needs_mask\s*=\s*bool\((.*?)\)\n", val, re.S)
    assert block, "could not find the geofix_needs_mask expression"
    named = set(re.findall(r"geofix_(\w+)", block.group(1)))
    for route in ("mask_in_camera", "mask_in_channels", "bridge_mask_noise",
                  "blend_train"):
        assert route in named, (
            f"geofix_{route} consumes the mask but is not in geofix_needs_mask; "
            "validation would leave the mask unbound for that arm.")
