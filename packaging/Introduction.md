# DiffSinger for TuneLab

A [DiffSinger](https://github.com/openvpi/DiffSinger)-based singing voice synthesis engine for TuneLab. It reads DiffSinger voicebanks in the **standard community format** — a model folder containing `dsconfig.yaml` + character metadata + predictor subdirectories — directly, with no conversion or repackaging.

> **Windows x64 only.** Optional GPU acceleration (DirectML, works with most discrete/integrated GPUs); falls back to CPU when no GPU is available. Windows on ARM is not supported yet — the host itself does not ship a win-arm64 build.

---

## 1. Installing models (voicebanks)

The plugin **ships with no models** — you supply your own. Default scan directory:

```
%APPDATA%\DiffSingerForTuneLab\Voices
```

(i.e. `C:\Users\<you>\AppData\Roaming\DiffSingerForTuneLab\Voices` — created automatically on first launch.)

Drop the **whole model folder** in, e.g.:

```
Voices\
└─ MyModel\
   ├─ dsconfig.yaml          ← acoustic config (required)
   ├─ character.yaml         ← character metadata (character.yaml OR character.txt, required)
   ├─ acoustic.onnx
   ├─ dsdur\ dspitch\ dsvariance\ …   ← predictor subdirectories
   ├─ (optional) dsvocoder\   ← vocoder shipped with the model, see §2
   └─ (optional) tunelab.yaml ← TuneLab-specific descriptor, see §5
```

Detection rule: a folder that contains **both `dsconfig.yaml` and `character.yaml` (or `character.txt`)** is recognized as a model. Folders may be nested (the scan recurses); once a model is found its subfolders are not descended into.

### Using other directories

Open **Settings → Extensions → DiffSinger** and add folders to **"Voicebank directories"** (one per row). The default directory is always active; the ones you add are **appended**. Changes trigger an immediate rescan — no restart needed.

---

## 2. Installing a vocoder

A DiffSinger acoustic model outputs a mel-spectrogram; a **vocoder** is required to turn it into audio. The vocoder is resolved in two steps, in this order (same rule as OpenUtau):

**① Bundled with the model — wins unconditionally.** If the model folder contains `dsvocoder\vocoder.yaml`, that vocoder is used and the `vocoder` field in `dsconfig.yaml` is **ignored**:

```
MyModel\
├─ dsconfig.yaml
└─ dsvocoder\           ← if this exists, it is used — install nothing
   ├─ vocoder.yaml
   └─ <model>.onnx
```

This is common in practice: `dsconfig.yaml` is often copied from a template with its `vocoder:` line left untouched, while the voicebank ships its own fine-tuned vocoder in `dsvocoder\`. Such models work out of the box — you do not need to install anything, and the name mismatch is harmless.

**② Otherwise, by folder name in the vocoder directories.** Default vocoder directory:

```
%APPDATA%\DiffSingerForTuneLab\Vocoders\<vocoder-name>\
   ├─ vocoder.yaml
   └─ <model>.onnx
```

Here `<vocoder-name>` must **match** the `vocoder` field in the model's `dsconfig.yaml` (case-sensitive). One vocoder can be shared by many models — install it once.

Vocoders can also live elsewhere: open **Settings → Extensions → DiffSinger** and add folders to **"Vocoder directories"** (one per row). The default directory is always active; the ones you add are **appended** and searched in order.

> If synthesis produces **no sound**, the model has no bundled `dsvocoder\` *and* no matching vocoder was found — install one whose folder name equals `dsconfig.yaml`'s `vocoder`, or drop the vocoder into the model's own `dsvocoder\`.

---

## 3. Settings

**Settings → Extensions → DiffSinger**:

| Setting | Description | Default |
|---|---|---|
| **Voicebank directories** | Extra model scan dirs (one per row) | empty (default dir only) |
| **Vocoder directories** | Extra vocoder scan dirs (one per row) | empty (default dir only) |
| **Execution device** | `GPU (DirectML)` or `CPU`. GPU is noticeably faster; use CPU if the driver misbehaves or you have no GPU | GPU (DirectML) |
| **Inference mode** | `Isolated process` runs ONNX in a separate process so a native crash can't take down TuneLab (auto-falls back to in-process if it can't start, e.g. blocked by antivirus); `In-process` runs inside TuneLab | Isolated process |
| **Sampling steps** | Diffusion sampling steps. Higher = finer but slower; 20 is usually enough | 20 |
| **Tensor cache** | Caches inference intermediates — repeated synthesis of the same segment is faster and reproducible | on |
| **Cache size limit (MB)** | Disk cap for the tensor cache; `0` = unlimited | 4096 |

---

## 4. Parameters

Which tracks show up in the parameter panel depends on what the selected model actually accepts — a track
exists only if the model has that input. Nothing here has to be touched to get sound: an untouched track has
no effect.

### 4.1 Energy / Breathiness / Voicing / Tension

Each of these has **two editable tracks plus one read-only curve**:

| In the UI | What it is |
|---|---|
| **Energy** (bottom tab bar) | *Offset* track. Relative: "whatever the model does here, make it a bit louder / breathier". The everyday tuning entry — being relative, it keeps working after you switch model / version or re-roll a take. |
| **Energy: actual** (bottom tab bar) | *Actual value* track, in real acoustic units (dB; Tension uses the model's own unit). Wherever you draw, **that value is what the model receives** — nothing is added on top. Stretches you leave alone fall back to the model's own output, so you can take over one syllable and nothing else. |
| **Energy** (parameter title bar chip) | Read-only curve: what the model actually received, drawn as a translucent area. |

The read-only curve and the *actual* track share one identity, which enables the move worth knowing:
light up **Energy: actual** and its read-only curve lights up with it — then right-click a range and **bake**
the model's curve into the track. You are now holding the model's own line as ordinary anchors and can reshape
one stretch of it while the rest stays exactly as the model made it.

Order of application: **model output → offset → actual override**. So baking and changing nothing sounds
identical, and wherever you drew on the *actual* track, that is precisely the value used.

> The *actual* track and the read-only curve only exist if the model's variance predictor actually predicts
> that parameter. Without a predictor the *actual* track is still editable — it then defines the value outright.

### 4.2 Other tracks

| Track | Appears when | Meaning |
|---|---|---|
| **Gender** | model takes a key-shift input | Formant shift. `0` = untouched, positive = shifted down. |
| **Speed** | model takes a speed input | Phoneme timing multiplier, `1` = as written. |
| **Mouth opening** | model takes a mouth-opening offset | `0` = no intervention. |
| **Expressiveness** | pitch predictor exports `expr` | `1` = the model's own pitch contour, `0` = stick to the written notes. |
| **Tone shift** | pitch-controllable vocoder | Semitones. The pitch you hear stays put; timbre is taken from another register — "sing this note with the voice of a higher / lower range". |
| **Pitch / Variance / Timbre seed** | `retake` declared in `tunelab.yaml` (§5) | Which *take*, per frame. `0` = keep the original take; anything you paint = re-roll just those frames. The seed is an identity, not an amount — `0.4` is not "more" than `0.3`, it is a different take. Values live in the project file, so a take survives closing and reopening. |
| **Speaker mix** | more than one singer exposed | One track per candidate singer, added from the part properties. Per-frame weights are normalized. |
| **Phoneme mix** | `phoneme_mix` declared (§6) | Frame-level blend towards another phoneme — see §6. |

### 4.3 Part / note / phoneme properties

- **Part**: Model and Version dropdowns (only when the model declares more than one), default Language,
  the Speaker mix container, and the Phoneme mix slot count.
- **Note**: Language override.
- **Phoneme**: Language override, plus the mix target of each slot.

---

## 5. `tunelab.yaml` (optional)

A model folder **may** include a `tunelab.yaml` carrying the "author decision layer" that the base voicebank format can't express but TuneLab wants. **It is entirely optional** — without it a model still works, loaded the default way (identical to how it behaves without this file: voice id = folder name, one voice per model, speakers via a dropdown).

What it enables:

- **Stable model / voice identity** — decoupled from the folder name, survives renames;
- **Splitting speakers into selectable singers** + a whitelist (expose only what you want);
- **Merging one person across multiple models** into a single top-level entry (data-retrain upgrades);
- **Versioning** — multiple versions of the same lineage, auto-follow-latest or explicitly pinned;
- **Retake capability declaration** — note-level pitch / variance / timbre retake requires the model to be built with the [externalized-noise build of DiffSinger](https://github.com/LiuYunPlayer/DiffSinger) (it exposes the diffusion noise as a `noise` input); standard exports cannot retake. Pitch / variance retake needs only **re-exporting** with it, while **acoustic (timbre) retake additionally requires retraining** with it. Declare only what your model actually supports;
- **Phoneme-mix capability declaration** — frame-level phoneme mixing (morphing a sound between phonemes over time, see §6) requires the acoustic + pitch + variance models to be **re-exported** with a build that supports it; declare `phoneme_mix: true` after re-exporting to expose the feature;
- **Localization** — model / singer / language names shown per the host language.

Minimal example (`<model-folder>/tunelab.yaml`):

```yaml
format: tunelab-voicebank/1
id: myteam.my-model            # stable model id (merge key)
name: My Model                 # display name
name_i18n: { zh-CN: 我的模型 }  # optional: localized name
version: 1
released: 2026-01              # optional: cross-model ordering (defines "latest")

retake:                        # optional: only set true if the model truly supports it
  pitch: true
  variance: false
  acoustic: false

phoneme_mix: true              # optional: all three models re-exported for frame-level phoneme mixing (see §6)

voices:                        # optional: presence = whitelist; absent = one voice per model
  - { id: singer-a, speaker: spk_a, name: Singer A, name_i18n: { zh-CN: 歌手 A } }
  - { id: singer-b, speaker: spk_b, name: Singer B }
```

Field reference:

| Field | Required | Notes |
|---|---|---|
| `format` | ✅ | fixed `tunelab-voicebank/1` |
| `id` | ✅ | stable model id; two models sharing a `voices[].id` are treated as the same person and merged |
| `name` / `name_i18n` | `name` ✅ | model display name + localization (keys `en-US` / `zh-CN`) |
| `version` | | version number within a model (integer, higher = newer) |
| `version_label` | | human-readable version label (display only) |
| `released` | | `YYYY` / `YYYY-MM` / `YYYY-MM-DD`, used for cross-model ordering |
| `retake.{pitch,variance,acoustic}` | | declares retake support; all default `false` (not exposed). A wrong declaration won't crash — synthesis silently treats it as unsupported |
| `phoneme_mix` | | declares frame-level phoneme-mix support; default `false` (not exposed). A wrong declaration won't crash — silently treated as unsupported |
| `voices[]` | | exposed singer whitelist: `id`=global singer id, `speaker`=this model's dsconfig suffix, plus `name`/`name_i18n`/`default_language`/`portrait`/`color` |
| `languages` | | language display-name overlay + whitelist (`id` must match a dsconfig language key) |

> A parse failure (malformed file) won't make the model disappear — the plugin warns and falls back to default loading.

---

## 6. Phoneme mix (frame-level)

> Only appears when the voicebank's `tunelab.yaml` declares `phoneme_mix: true` (see §5); voicebanks without it show no related controls.

Morphs a sound **between phonemes over time** — e.g. an `a` gliding smoothly to `o` within a note — with timbre, pitch and variance moving together.

1. In the **part properties**, set **"Phoneme mix slots"** to ≥1 (`0` = off).
2. After one synthesis, each phoneme's property panel shows **"Mix phoneme k"** (target phoneme — type the bare symbol, e.g. `o`) + **"Mix language k"** (multi-language banks only; defaults to following this phoneme's language).
3. The automation panel gets a **"Phoneme mix k"** curve — draw it to control **when / how much** (`0` = no mix, `1` = fully the target phoneme).
4. **Multiple slots**: raise the slot count to mix toward several targets at once (independent target + curve per slot); when per-frame slot weights sum above `1` they are auto-normalized, so it won't break up.

> Sweet spot: transitions **between same-family vowels** (a↔o, e↔i) sound most natural; cross-category (vowel↔consonant) is not guaranteed.

---

## 7. Troubleshooting

- **Model missing from the singer list** → verify the folder has **both** `dsconfig.yaml` and `character.yaml` (or `.txt`); confirm it's under the default dir or a dir you added in settings; settings changes auto-rescan.
- **No sound after synthesis** → see §2; usually a missing or misnamed vocoder.
- **Too slow** → set execution device to GPU (DirectML); or lower the sampling steps; keep the tensor cache on (repeated synthesis gets much faster).
- **Reproducing a previous render** → keep the tensor cache on; identical input hits the cache and reproduces the result.

---

License and third-party attributions are in the bundled `THIRD-PARTY-NOTICES.md`.
