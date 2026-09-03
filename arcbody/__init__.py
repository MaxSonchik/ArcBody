"""ArcBody — body-shape embeddings and anthropometry for GenAI conditioning.

ArcBody is the body-side counterpart to a face-recognition ArcFace service: the
same additive-angular-margin idea, applied to whole-body crops instead of faces.
It answers three questions about a person's *body*:

1. "What does this body look like, numerically?"  -> a 256-d L2-normalised
   embedding that is stable across photos, poses and outfits of one person.
2. "What are this body's measurements?"           -> anthropometry in cm, scaled
   by a client-supplied stature.
3. "Did the generator keep the body?"             -> cosine similarity between a
   reference profile and a generated image, plus per-measurement deltas.

Face identity is explicitly out of scope: ArcBody consumes an opaque
``person_id`` that the existing ArcFace service owns.  See ``docs/ARCHITECTURE.md``.
"""

from arcbody.version import __version__

__all__ = ["__version__"]
