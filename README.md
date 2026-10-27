# Two facing DOI modules: student starting workflow

This folder is self-contained. It copies the optical/material configuration and
optical response reduction from `hpc-27-09/stream_workflow`, with both light-guide
faces now pointing towards the central source. Each module has an 8x8 array of
3 mm wide LYSO pixels on a 3.2 mm pitch, a 1 mm front guide, 0.1 mm back glass,
and 0.5 mm SiPM layer. The default crystal length is 15 mm.

## Run locally

Activate your OpenGATE environment, then work inside this folder:

```bash
python simulate.py --events 1 --visu --output-dir visual-test
python simulate.py --events 100 --pixel-length-mm 15 --output-dir length-15
python analyse.py --input-dir length-15 --pde 0.30
```

`--events` is **per thread**. `--no-optical` is useful for quickly testing geometry
and scattering truth; PE signals will be zero. `--check-overlaps` enables the
geometry check. `--gap-mm 600` is the separation between the outer front-glass
faces. Changing crystal length leaves those faces fixed and extends the crystals
away from the source. Both modules use the same selected length.

Open `explore.ipynb` in your analysis environment (NumPy, pandas, Matplotlib).
Set `CSV` to the run's `events.csv`. Simulation/reduction additionally need
OpenGATE and uproot. Use the same PDE and optical properties for length comparisons.
PE counts are an energy proxy, not calibrated keV.

## Student projects

1. Vary crystal length and study DOI reconstruction using the per-pixel PE images
   and true first-interaction depths.
2. Study scattering reconstruction and event matching, then implement a model
   of the entanglement effect by reweighting the baseline events using the true
   Compton angles and scattering-plane separation. Event weights are not yet
   implemented. The chosen weighting prescription and its normalization must be
   validated against its assumptions and the event selections.

## Source and limits of the physics study

The inherited efficient source emits exactly back-to-back 511-keV photons and
an independent isotropic 1274.5-keV photon from the origin. Pair directions are
uniform in solid angle within a rectangular cone that intersects both arrays.
The source has a provisional PMMA capsule (radius 12.7 mm, thickness 6.35 mm).
Capsule and air interactions are transported normally. This is a conditional
Na-22-like photon source, not a full radioactive-decay/positron simulation.
It cannot supply absolute isotropic-source efficiency without acceptance accounting.
The run configuration records the geometric acceptance fraction.

**This source does not encode entangled photon polarizations.** The baseline can
study first/second interaction identification, containment, angular reconstruction,
and matching geometry. An entanglement prediction requires a separately validated model: for the student
project this is planned as truth-angle event reweighting, with the baseline
physics and angular conventions taken into account. The recorded
`polarization_entanglement_modelled: false` must not be interpreted otherwise.
Inherited LYSO composition is the existing LSO-like approximation, not a new
material validation. Optical transport uses the existing 100 ns late-photon kill.

## Event table

`events.csv` contains one row per generated event, including events undetected in
one or both modules. `simulation_id`, `RunID`, `EventID` identify its provenance;
use unique output directory names when combining runs. Event IDs are read from
the source-primary records because GATE worker IDs can have gaps. No energy/PE cuts are
applied by this analysis. `both_detected` means at least one sampled PE in each
module **in the same generated event**, not a continuous-stream timing coincidence.
Random coincidences, electronic noise and timing smearing are not modelled here.

Module prefixes `A_` and `B_` distinguish identical sets of columns:

| Columns | Meaning |
| --- | --- |
| `npe_0_0` ... `npe_7_7` | 64 measured PE counts, ordered `(local x index, local y index)` |
| `npe_total`, `max_npe`, `max_ix`, `max_iy` | Signal summary; maximum pixel indices absent if undetected |
| `w` | Maximum PE divided by total PE; undefined for zero signal |
| `first_pe_time_ns` | First detected optical-photon global time, if present |
| `truth_total_edep_kev` | Total non-optical deposited energy in that module's LYSO, including secondaries and prompt gamma |
| `truth_n_primary_tracks` | Number of original 511-keV gamma tracks interacting in this module |
| `truth_n_interactions` | Number of physical LYSO steps of the selected primary track |
| `truth_i1_*`, `truth_i2_*` | First and second physical LYSO interaction of the same selected primary 511-keV track |
| `truth_compton_angle_deg` | Angle between incoming/outgoing directions at first CS |
| `truth_two_site_angle_deg` | Angle using incoming direction and first-to-second site displacement |
| `truth_energy_angle_deg` | Compton formula using true before/after gamma kinetic energies |
| `truth_site_separation_mm` | Distance between recorded first and second sites |

Each `truth_i1_*` / `truth_i2_*` group contains `type` (`PE`, `CS`, `Rayleigh`,
`other`), raw `process`, `track_id`, `pixel`, `time_ns`, `world_x/y/z_mm`,
`local_x/y/z_mm`, `depth_mm`, `incoming_x/y/z`, `outgoing_x/y/z`,
`energy_before_kev`, `energy_after_kev`, `energy_transfer_kev`, and `local_edep_kev`.
Directions are world-frame unit vectors. Gamma energy transfer and local step
energy deposition differ: energy carried away by an electron is not deposited
locally on the gamma step. Do not filter physical interactions by nonzero edep.

Local z runs from glass to SiPM in **both** modules. `local_z_mm` ranges from
-L/2 to +L/2; `depth_mm = local_z_mm + L/2` ranges from 0 to L. Module B is
rotated by 180 degrees about world y, so its local x and z point opposite A's.
Missing truth remains NaN, never a fabricated interaction at zero depth.

Primary tracks are identified by ParentID=0 and their vertex energy, so a gamma
scattered in the capsule is still tracked. Only interactions inside LYSO are
numbered; the first LYSO interaction need not be the first interaction since
emission. If multiple original 511-keV tracks interact in one module, the one
with the earliest physical interaction is selected and the count flags ambiguity.
Rayleigh is retained in the sequence. The two stored sites may be in the same
pixel or separated by transport/scattering outside LYSO, so the two-site angle
need not equal the direct first-scatter angle.

`truth_distinct_pair` requires one original track in each module and distinct
track IDs. `truth_plane_angle_deg` is the **folded, unoriented** angle between
first CS planes (0 to 90 degrees), using world-frame track directions. It is
only defined for a distinct pair with first CS in both modules. It is not a
measured observable or evidence of entanglement.

The notebook makes explicit selections and simple plots: PE spectrum, W vs true
DOI, first/second interaction types, reconstructed vs true scattering angles,
and scattering-plane separation. These are exploratory, not performance claims.

## Cluster pilot

From this directory on Sonic:

```bash
sbatch submit.sh
# Override a length/workload when ready:
PIXEL_LENGTH_MM=20 EVENTS_PER_THREAD=1000 sbatch submit.sh
```

The script runs simulation and analysis in each task, retaining ROOT and CSV files.
It defaults to two tasks, two CPUs each, 100 events/thread, and two hours. Tune
resources after measuring a pilot. Output goes to a separate scratch directory.
To combine only this submission's CSVs:

```python
from pathlib import Path
import pandas as pd
base = Path('/scratch/keaveneyj/student-two-module')
files = sorted(base.glob('job_JOBID_*/events.csv'))  # replace JOBID
pd.concat([pd.read_csv(p) for p in files], ignore_index=True).to_csv('events-merged.csv', index=False)
```

## Checks performed during setup

- A 50-event, 10 mm, no-optical run completed and reduced to 50 rows.
- A two-thread, 20 mm optical run completed with overlap checking: six generated
  events, six CSV rows, and two events detected in both modules.
- The test suite checks same-track interaction order, preservation of Rayleigh
  steps, exclusion of prompt/secondary tracks from the 511 truth labels, missing
  second interactions, coordinate conversion and a 90-degree Compton example.

Run the tests with `python -m unittest discover -s . -p 'test_*.py'`. These are
small implementation checks; physics and resolution still require validation
with larger samples.
