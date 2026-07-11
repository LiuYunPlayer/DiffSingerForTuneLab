# DiffSinger for TuneLab

A [DiffSinger](https://github.com/openvpi/DiffSinger)-based singing voice synthesis engine for TuneLab. It reads DiffSinger voicebanks in the **standard community format** — a model folder containing `dsconfig.yaml` + character metadata + predictor subdirectories — directly, with no conversion or repackaging.

> **Windows only.** Optional GPU acceleration (DirectML, works with most discrete/integrated GPUs); falls back to CPU when no GPU is available.

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
   └─ (optional) tunelab.yaml ← TuneLab-specific descriptor, see §4
```

Detection rule: a folder that contains **both `dsconfig.yaml` and `character.yaml` (or `character.txt`)** is recognized as a model. Folders may be nested (the scan recurses); once a model is found its subfolders are not descended into.

### Using other directories

Open **Settings → Extensions → DiffSinger** and add folders to **"Voicebank directories"** (one per row). The default directory is always active; the ones you add are **appended**. Changes trigger an immediate rescan — no restart needed.

---

## 2. Installing a vocoder

A DiffSinger acoustic model outputs a mel-spectrogram; a **vocoder** is required to turn it into audio. Default vocoder directory:

```
%APPDATA%\DiffSingerForTuneLab\Vocoders\<vocoder-name>\
   ├─ vocoder.yaml
   └─ <model>.onnx
```

`<vocoder-name>` must **match** the `vocoder` field in the model's `dsconfig.yaml` (case-sensitive). One vocoder can be shared by many models — install it once.

Vocoders can also live elsewhere: open **Settings → Extensions → DiffSinger** and add folders to **"Vocoder directories"** (one per row). The default directory is always active; the ones you add are **appended** and searched in order.

> If synthesis produces **no sound**, the vocoder is most likely missing or its folder name doesn't match `dsconfig.yaml`'s `vocoder`.

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

## 4. `tunelab.yaml` (optional)

A model folder **may** include a `tunelab.yaml` carrying the "author decision layer" that the base voicebank format can't express but TuneLab wants. **It is entirely optional** — without it a model still works, loaded the default way (identical to how it behaves without this file: voice id = folder name, one voice per model, speakers via a dropdown).

What it enables:

- **Stable model / voice identity** — decoupled from the folder name, survives renames;
- **Splitting speakers into selectable singers** + a whitelist (expose only what you want);
- **Merging one person across multiple models** into a single top-level entry (data-retrain upgrades);
- **Versioning** — multiple versions of the same lineage, auto-follow-latest or explicitly pinned;
- **Retake capability declaration** — note-level pitch / variance / timbre retake requires the model to be built with the [externalized-noise build of DiffSinger](https://github.com/LiuYunPlayer/DiffSinger) (it exposes the diffusion noise as a `noise` input); standard exports cannot retake. Pitch / variance retake needs only **re-exporting** with it, while **acoustic (timbre) retake additionally requires retraining** with it. Declare only what your model actually supports;
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
| `voices[]` | | exposed singer whitelist: `id`=global singer id, `speaker`=this model's dsconfig suffix, plus `name`/`name_i18n`/`default_language`/`portrait`/`color` |
| `languages` | | language display-name overlay + whitelist (`id` must match a dsconfig language key) |

> A parse failure (malformed file) won't make the model disappear — the plugin warns and falls back to default loading.

---

## 5. Troubleshooting

- **Model missing from the singer list** → verify the folder has **both** `dsconfig.yaml` and `character.yaml` (or `.txt`); confirm it's under the default dir or a dir you added in settings; settings changes auto-rescan.
- **No sound after synthesis** → see §2; usually a missing or misnamed vocoder.
- **Too slow** → set execution device to GPU (DirectML); or lower the sampling steps; keep the tensor cache on (repeated synthesis gets much faster).
- **Reproducing a previous render** → keep the tensor cache on; identical input hits the cache and reproduces the result.

---

License and third-party attributions are in the bundled `THIRD-PARTY-NOTICES.md`.
