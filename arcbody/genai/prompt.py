"""Turning a body profile into text a generator can act on.

Three rules shape what comes out.

**Only say what is known.** A measurement whose confidence interval is wider
than the configured tolerance is dropped rather than rounded. A generator given
"waist 84 cm" when the estimate was 84 +/- 15 will render one specific wrong
body with total conviction; given nothing, it renders a plausible one.

**Say proportions, not just numbers.** Diffusion models respond far better to
"broad shoulders tapering to a narrow waist" than to three integers, because
that is the language their captions were written in. Numbers are included for
the pipelines that parse them, phrasing for the ones that read.

**Never describe the person.** Age, sex, ethnicity and attractiveness are not
measured here and are not guessed. The output describes a *body's geometry*, and
anything beyond that is the caller's to supply — which also keeps ArcBody out of
the business of inferring protected attributes from a photograph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arcbody.config import GenAISettings
from arcbody.measure.calibration import band_of
from arcbody.types import BodyMeasurements

#: Wording for each end of a ratio's population range. Five bands, because three
#: is too coarse to distinguish a build and seven invents precision.
_DESCRIPTORS: dict[str, tuple[str, str, str, str, str]] = {
    "shoulder_to_stature": (
        "very narrow shoulders",
        "narrow shoulders",
        "average shoulder width",
        "broad shoulders",
        "very broad shoulders",
    ),
    "waist_to_stature": (
        "a very slim waist",
        "a slim waist",
        "an average waist",
        "a full waist",
        "a very full waist",
    ),
    "hip_to_stature": (
        "very narrow hips",
        "narrow hips",
        "average hips",
        "wide hips",
        "very wide hips",
    ),
    "chest_to_stature": (
        "a very narrow chest",
        "a narrow chest",
        "an average chest",
        "a broad chest",
        "a very broad chest",
    ),
    "leg_to_stature": (
        "very short legs",
        "short legs",
        "averagely proportioned legs",
        "long legs",
        "very long legs",
    ),
    "torso_to_stature": (
        "a very short torso",
        "a short torso",
        "an average torso",
        "a long torso",
        "a very long torso",
    ),
    "thigh_to_stature": (
        "very slender thighs",
        "slender thighs",
        "average thighs",
        "muscular thighs",
        "very muscular thighs",
    ),
    "upper_arm_to_stature": (
        "very slender arms",
        "slender arms",
        "average arms",
        "muscular arms",
        "very muscular arms",
    ),
}

#: Ratios worth mentioning, in the order a caption would naturally use.
_NARRATIVE_ORDER = (
    "shoulder_to_stature",
    "chest_to_stature",
    "waist_to_stature",
    "hip_to_stature",
    "torso_to_stature",
    "leg_to_stature",
    "thigh_to_stature",
    "upper_arm_to_stature",
)

#: Measurements quoted numerically, in the order a tailor would list them.
_NUMERIC_ORDER = (
    "stature",
    "shoulder_breadth",
    "chest_girth",
    "waist_girth",
    "hip_girth",
    "inseam",
    "arm_length",
    "thigh_girth",
    "neck_girth",
)

_LABELS: dict[str, str] = {
    "stature": "height",
    "shoulder_breadth": "shoulder width",
    "chest_girth": "chest",
    "waist_girth": "waist",
    "hip_girth": "hips",
    "inseam": "inseam",
    "arm_length": "arm length",
    "thigh_girth": "thigh",
    "neck_girth": "neck",
}


@dataclass
class PromptBundle:
    """Text in three forms, because downstream pipelines want different ones."""

    prompt: str
    build_phrase: str
    proportion_phrases: list[str] = field(default_factory=list)
    measurement_phrases: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "build_phrase": self.build_phrase,
            "proportion_phrases": self.proportion_phrases,
            "measurement_phrases": self.measurement_phrases,
            "omitted_for_low_confidence": self.omitted,
        }


def _band(name: str, value: float) -> int | None:
    """Which descriptor band a ratio falls in, against the fitted reference."""
    return band_of(name, value)


def describe_ratio(name: str, value: float) -> str | None:
    """A phrase for one proportion, or ``None`` if it is unremarkable."""
    words = _DESCRIPTORS.get(name)
    band = _band(name, value)
    if words is None or band is None:
        return None
    # The middle band is "average", which adds nothing to a prompt and dilutes
    # the phrases that do carry signal.
    return None if band == 2 else words[band]


def _format_length(value_cm: float, units: str) -> str:
    if units == "imperial":
        return f"{value_cm / 2.54:.0f} in"
    if units == "both":
        return f"{value_cm:.0f} cm ({value_cm / 2.54:.0f} in)"
    return f"{value_cm:.0f} cm"


def build(
    measurements: BodyMeasurements,
    settings: GenAISettings | None = None,
    *,
    subject: str = "a person",
) -> PromptBundle:
    """Compose the prompt bundle for one body."""
    settings = settings or GenAISettings()

    proportions = [
        phrase
        for phrase in (
            describe_ratio(name, measurements.ratios[name])
            for name in _NARRATIVE_ORDER
            if name in measurements.ratios
        )
        if phrase
    ]

    numeric: list[str] = []
    omitted: list[str] = []
    confident = measurements.confident_values(settings.prompt_max_relative_ci)
    for name in _NUMERIC_ORDER:
        value = measurements.values.get(name)
        if value is None:
            continue
        if name not in confident:
            omitted.append(name)
            continue
        numeric.append(
            f"{_LABELS.get(name, name)} {_format_length(value.value_cm, settings.units)}"
        )

    build_phrase = measurements.somatotype or "average build"
    if measurements.stature_cm and measurements.weight_kg:
        from arcbody.measure.anthropometry import body_mass_index

        bmi = body_mass_index(measurements.stature_cm, measurements.weight_kg)
        build_phrase = f"{build_phrase}, BMI {bmi:.0f}"

    parts = [f"{subject} with {build_phrase}"]
    if proportions:
        parts.append(_join(proportions))
    if numeric:
        parts.append("measurements: " + ", ".join(numeric))
    prompt = "; ".join(parts)

    return PromptBundle(
        prompt=prompt,
        build_phrase=build_phrase,
        proportion_phrases=proportions,
        measurement_phrases=numeric,
        omitted=omitted,
    )


def _join(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def negative_prompt(measurements: BodyMeasurements) -> str:
    """Terms to steer away from, derived from the body's own extremes.

    Generators regress towards their training distribution's average body, so
    the useful negative for an unusual build is the opposite of what makes it
    unusual — naming it explicitly is what keeps a broad-shouldered subject from
    coming back average.
    """
    opposites = {
        0: "narrow",
        1: "narrow",
        3: "wide",
        4: "wide",
    }
    terms: list[str] = []
    for name in ("shoulder_to_stature", "waist_to_stature", "hip_to_stature"):
        value = measurements.ratios.get(name)
        if value is None:
            continue
        band = _band(name, value)
        if band in (0, 1):
            terms.append(f"{opposites[3]} {name.split('_')[0]}")
        elif band in (3, 4):
            terms.append(f"{opposites[0]} {name.split('_')[0]}")
    terms.append("distorted proportions")
    terms.append("extra limbs")
    return ", ".join(dict.fromkeys(terms))
