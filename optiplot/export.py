"""Portable reproducibility bundles with data, recipe, renderer and figures."""

from pathlib import Path
from dataclasses import asdict
import hashlib
import io
import json
import zipfile
import importlib.metadata
import matplotlib
from .render import render
from .style import Style


def export_bundle(profile, recommendation, path, options=None):
    options = options or {}
    # Resolved once here purely so the recipe records the full style; render()
    # splits the same dict again on its own.
    style, _ = Style.from_options(options)
    data = profile.data.to_csv(index=False).encode("utf-8-sig")
    recipe = {
        "format_version": 2,
        "recommendation": asdict(recommendation),
        # The whole resolved style, not just what the caller happened to pass:
        # a replay must not depend on which defaults the current version ships.
        "style": style.to_dict(),
        "options": options,
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "source_name": Path(profile.path).name,
        "notes": profile.notes,
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "matplotlib"]},
    }
    fig = render(profile, recommendation, options=options)
    replay = """from pathlib import Path
from types import SimpleNamespace
import json, hashlib
import pandas as pd
from rendering import render
from style import Style
root = Path(__file__).resolve().parent
recipe = json.loads((root / 'recipe.json').read_text(encoding='utf-8'))
raw = (root / 'data.csv').read_bytes()
if hashlib.sha256(raw).hexdigest() != recipe['data_sha256']:
    raise ValueError('Data checksum differs from export recipe')
profile = SimpleNamespace(data=pd.read_csv(root / 'data.csv'))
rec = SimpleNamespace(**recipe['recommendation'])
# style.json is also shipped standalone so the look can be reused as a preset
style = Style.load_preset(root / 'style.json')
for suffix in ['png', 'svg', 'pdf']:
    render(profile, rec, root / ('reproduced.' + suffix),
           options=recipe['options'], style=style)
print('Reproduced PNG, SVG and PDF')
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.csv", data)
        z.writestr("recipe.json", json.dumps(recipe, ensure_ascii=False, indent=2))
        z.writestr(
            "style.json", json.dumps(style.to_dict(), ensure_ascii=False, indent=2) + "\n"
        )
        z.writestr("render_plot.py", replay)
        z.writestr(
            "rendering.py", Path(__file__).with_name("render.py").read_text(encoding="utf-8-sig")
        )
        # render.py imports Style; the bundle has to carry it or replay fails.
        z.writestr(
            "style.py", Path(__file__).with_name("style.py").read_text(encoding="utf-8-sig")
        )
        z.writestr(
            "requirements.txt", "\n".join(f"{p}=={v}" for p, v in recipe["versions"].items())
        )
        z.writestr(
            "README.txt",
            "Unzip; pip install -r requirements.txt; python render_plot.py\nThe table preview is limited; data.csv contains all imported rows.\n",
        )
        for ext in ["png", "svg", "pdf"]:
            buf = io.BytesIO()
            with matplotlib.rc_context(style.rc_params()):
                fig.savefig(buf, format=ext, **style.savefig_kwargs())
            z.writestr("figure." + ext, buf.getvalue())
    fig.clear()
    return Path(path)
