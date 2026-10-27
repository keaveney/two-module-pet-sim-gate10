"""Facing DOI modules with an acceptance-conditioned Na-22-like source."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
N_PIXELS = 8
PIXEL_MM = np.array([3.0, 3.0, 15.0])
PITCH_MM = 3.2
ARRAY_MM = N_PIXELS * 3.0 + (N_PIXELS - 1) * 0.2
BLOCK_MM = ARRAY_MM + 2 * 0.2
GLASS_MM = 1.0
BACK_GLASS_MM = 0.1
SIPM_MM = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=1,
                        help="Acceptance-conditioned source events per thread")
    parser.add_argument("--pixel-length-mm", type=float, default=15.0,
                        help="Longitudinal LYSO length in both modules")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=HERE / "local-test")
    parser.add_argument("--gap-mm", type=float, default=600.0)
    parser.add_argument("--visu", action="store_true", help="Open the Qt viewer; requires one thread")
    parser.add_argument("--no-optical", action="store_true", help="Disable optical physics for geometry/decay checks")
    parser.add_argument("--check-overlaps", action="store_true")
    parser.add_argument("--capsule-radius-mm", type=float, default=12.7)
    parser.add_argument("--capsule-thickness-mm", type=float, default=6.35)
    args = parser.parse_args()
    if min(args.events, args.threads, args.gap_mm, args.pixel_length_mm,
           args.capsule_radius_mm, args.capsule_thickness_mm) <= 0:
        parser.error("Counts and dimensions must be positive.")
    if args.visu and args.threads != 1:
        parser.error("Use --threads 1 with --visu.")
    if args.capsule_thickness_mm >= args.gap_mm:
        parser.error("The capsule must fit between the modules.")
    PIXEL_MM[2] = args.pixel_length_mm
    return args


def module_layout(gap_mm):
    """World z joins the modules. Each local z runs glass -> SiPM."""
    return {
        "A": {
            "centre_mm": np.array([0.0, 0.0, gap_mm / 2 + PIXEL_MM[2] / 2 + GLASS_MM]),
            "rotation": np.eye(3),
        },
        "B": {
            "centre_mm": np.array([0.0, 0.0, -gap_mm / 2 - PIXEL_MM[2] / 2 - GLASS_MM]),
            # Ry(180 degrees): both glass faces point towards the source.
            "rotation": np.diag([-1.0, 1.0, -1.0]),
        },
    }


def add_module(sim, label, placement, u, optical):
    centre = placement["centre_mm"]
    rotation = placement["rotation"]

    def box(suffix, size_mm, local_centre_mm, material, color):
        volume = sim.add_volume("Box", f"{label}_{suffix}")
        volume.mother = "world"
        volume.size = (np.asarray(size_mm) * u.mm).tolist()
        volume.translation = ((centre + rotation @ np.asarray(local_centre_mm)) * u.mm).tolist()
        volume.rotation = rotation
        volume.material = material
        volume.color = color
        return volume.name

    pixels = []
    offset = (N_PIXELS - 1) * PITCH_MM / 2
    for ix in range(N_PIXELS):
        for iy in range(N_PIXELS):
            pixels.append(box(f"pixel_{ix}_{iy}", PIXEL_MM,
                              [ix * PITCH_MM - offset, iy * PITCH_MM - offset, 0],
                              "LYSO", [1.0, 0.2, 0.2, 0.55]))

    front = box("front_glass", [BLOCK_MM, BLOCK_MM, GLASS_MM],
                [0, 0, -PIXEL_MM[2] / 2 - GLASS_MM / 2],
                "G4_SILICON_DIOXIDE", [0.1, 0.8, 0.2, 0.7])
    back = box("back_glass", [BLOCK_MM, BLOCK_MM, BACK_GLASS_MM],
               [0, 0, PIXEL_MM[2] / 2 + BACK_GLASS_MM / 2],
               "G4_SILICON_DIOXIDE", [0.9, 0.1, 0.9, 0.7])
    sipm = box("sipm_block", [BLOCK_MM, BLOCK_MM, SIPM_MM],
               [0, 0, PIXEL_MM[2] / 2 + BACK_GLASS_MM + SIPM_MM / 2],
               "G4_Si", [0.1, 0.3, 1.0, 1.0])

    # Preserve the original pixel side-reflector and end-face treatment.
    # The back glass is otherwise exposed to world along its perimeter, allowing
    # optical photons to escape laterally before reaching the SiPM block.
    if optical:
        for pixel in pixels:
            for a, b in [(pixel, "world"), ("world", pixel)]:
                sim.physics_manager.add_optical_surface(a, b, "ESR_fast_reflector")
            sim.physics_manager.add_optical_surface(pixel, front, "Diffuse_front")
            sim.physics_manager.add_optical_surface(front, pixel, "Diffuse_front")
        sim.physics_manager.add_optical_surface(front, "world", "ESR_fast_reflector")
        for a, b in [(back, "world"), ("world", back)]:
            sim.physics_manager.add_optical_surface(a, b, "ESR_fast_reflector")
    return {"pixels": pixels, "front": front, "back": back, "sipm": sipm}


def conditional_pair_acceptance(layout):
    """Directions from the point source that cross active LYSO in both modules.

    The returned cone is the exact intersection of the two rectangular active
    areas for this centred geometry.  It conditions the pair direction only;
    transport inside the capsule and detector remains normal Geant4 physics.
    """
    a = layout["A"]
    b = layout["B"]
    offset = (N_PIXELS - 1) * PITCH_MM / 2
    half_x = offset + PIXEL_MM[0] / 2
    half_y_a = offset + PIXEL_MM[1] / 2
    half_y_b = half_y_a
    source = np.zeros(3)
    a_face = local_to_world(np.array([0.0, 0.0, -PIXEL_MM[2] / 2]), a)
    b_face = local_to_world(np.array([0.0, 0.0, -PIXEL_MM[2] / 2]), b)
    a_distance = abs(a_face[2] - source[2])
    b_distance = abs(b_face[2] - source[2])
    slope_x = min(half_x / a_distance, half_x / b_distance)
    slope_y = min(half_y_a / a_distance, half_y_b / b_distance)
    # Solid angle of a centred rectangle on the plane z = 1, with half-sides
    # slope_x and slope_y. The fraction is for one labelled 511-keV gamma
    # travelling toward A; an unlabeled physical pair has two orientations.
    solid_angle = 4 * np.arctan(slope_x * slope_y / np.sqrt(1 + slope_x**2 + slope_y**2))
    return slope_x, slope_y, solid_angle / (4 * np.pi)


def local_to_world(local, placement):
    return placement["centre_mm"] + placement["rotation"] @ np.asarray(local)


def isotropic_directions(rng, count):
    z = rng.uniform(-1.0, 1.0, count)
    phi = rng.uniform(0.0, 2 * np.pi, count)
    r = np.sqrt(1 - z**2)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def write_conditional_phase_space(path, events_per_thread, threads, seed, layout):
    """Create acceptance-conditioned Na-22-like gamma triplets as a TTree.

    Each event is [1275 keV, 511 keV towards A, 511 keV towards B]. The first,
    high-energy row marks an event boundary for GatePhaseSpaceSource.
    """
    import uproot

    total_events = events_per_thread * threads
    slope_x, slope_y, direction_fraction = conditional_pair_acceptance(layout)
    rng = np.random.default_rng(seed)

    # Uniform solid-angle sampling in the narrow rectangular acceptance cone.
    # The rejection factor is below one and is almost unity here, but retains
    # the proper dOmega/du/dv Jacobian rather than sampling slopes uniformly.
    u_values, v_values = [], []
    while len(u_values) < total_events:
        n = max(1024, 2 * (total_events - len(u_values)))
        u = rng.uniform(-slope_x, slope_x, n)
        v = rng.uniform(-slope_y, slope_y, n)
        keep = rng.random(n) < (1 + u**2 + v**2) ** -1.5
        u_values.extend(u[keep].tolist())
        v_values.extend(v[keep].tolist())
    u = np.asarray(u_values[:total_events])
    v = np.asarray(v_values[:total_events])
    toward_a = np.column_stack((u, v, np.ones(total_events)))
    toward_a /= np.linalg.norm(toward_a, axis=1, keepdims=True)
    toward_b = -toward_a
    prompt_gamma = isotropic_directions(rng, total_events)

    directions = np.empty((3 * total_events, 3), dtype=np.float32)
    directions[0::3] = prompt_gamma
    directions[1::3] = toward_a
    directions[2::3] = toward_b
    energy = np.empty(3 * total_events, dtype=np.float32)
    energy[0::3] = 1.2745
    energy[1::3] = 0.51099895
    energy[2::3] = 0.51099895
    zeros = np.zeros(3 * total_events, dtype=np.float32)
    branches = {
        "Position_X": zeros, "Position_Y": zeros, "Position_Z": zeros,
        "Direction_X": directions[:, 0], "Direction_Y": directions[:, 1], "Direction_Z": directions[:, 2],
        "KineticEnergy": energy,
        "PDGCode": np.full(3 * total_events, 22, dtype=np.int32),
        "Weight": np.ones(3 * total_events, dtype=np.float32),
    }
    with uproot.recreate(path) as root_file:
        # GatePhaseSpaceSource expects a TTree. Direct uproot assignment would
        # create an RNTuple in current uproot releases and is not compatible.
        tree = root_file.mktree("conditional_na22", {name: value.dtype for name, value in branches.items()})
        tree.extend(branches)
    return {
        "phase_space_file": str(path),
        "conditional_direction_acceptance_fraction": direction_fraction,
        "conditional_pair_acceptance_fraction": 2 * direction_fraction,
        "conditional_slope_x": slope_x,
        "conditional_slope_y": slope_y,
        "conditional_events": total_events,
    }


def add_source(sim, args, layout):
    """Add the acceptance-conditioned Na-22-like three-gamma source."""
    from opengate.utility import g4_units as u

    phase_space = args.output_dir / "conditional_na22_source.root"
    metadata = write_conditional_phase_space(
        phase_space, args.events, args.threads, args.seed, layout
    )
    source = sim.add_source("PhaseSpaceSource", "conditional_na22")
    source.attached_to = "world"
    source.phsp_file = str(phase_space)
    source.global_flag = True
    source.particle = ""
    source.position_key = "Position"
    source.direction_key = "Direction"
    source.energy_key = "KineticEnergy"
    source.PDGCode_key = "PDGCode"
    source.weight_key = "Weight"
    source.n = args.events
    source.batch_size = max(3, min(30_000, 3 * args.events))
    # GATE accepts a per-thread entry list only in MT mode.  A scalar is
    # required for the one-thread local visualisation/check workflow.
    source.entry_start = (
        0 if args.threads == 1
        else [3 * args.events * thread for thread in range(args.threads)]
    )
    source.generate_until_next_primary = True
    source.primary_PDGCode = 22
    source.primary_lower_energy_threshold = 1.0 * u.MeV
    metadata["source"] = (
        "Acceptance-conditioned Na-22-like event: isotropic 1275-keV gamma plus an exact "
        "back-to-back 511-keV pair constrained to cross the two detector-array apertures."
    )
    return metadata


def add_scoring(sim, modules, capsule_name, output_dir, optical, u):
    gamma_only = sim.add_filter("ParticleFilter", "gamma_only")
    gamma_only.particle = "gamma"
    optical_only = sim.add_filter("ParticleFilter", "optical_only")
    optical_only.particle = "opticalphoton"
    nonoptical = sim.add_filter("ParticleFilter", "nonoptical")
    nonoptical.particle = "opticalphoton"
    nonoptical.policy = "reject"

    ids = ["RunID", "EventID", "ThreadID", "TrackID", "ParentID"]
    gamma_attrs = ids + ["TrackCreatorProcess", "ProcessDefinedStep", "GlobalTime",
                         "PrePosition", "PostPosition", "PreDirection", "PostDirection",
                         "TrackVertexKineticEnergy",
                         "PreKineticEnergy", "PostKineticEnergy", "TotalEnergyDeposit",
                         "TrackVolumeName"]

    # A single volume NAME is important for GATE 10 entering/exiting actors.
    emitted = sim.add_actor("PhaseSpaceActor", "source_primaries")
    emitted.attached_to = capsule_name
    emitted.steps_to_store = "first"
    emitted.filters = [gamma_only]
    emitted.attributes = ids + ["TrackCreatorProcess", "PreKineticEnergy", "PostPosition", "PostDirection", "GlobalTime"]
    emitted.output_filename = str(output_dir / "source_primaries.root")

    late = sim.add_filter("ThresholdAttributeFilter", "late_optical")
    late.attribute = "TimeFromBeginOfEvent"
    late.value_min = 100 * u.ns

    for label, volumes in modules.items():
        gamma = sim.add_actor("DigitizerHitsCollectionActor", f"gamma_steps_{label}")
        gamma.attached_to = volumes["pixels"]
        gamma.attributes = gamma_attrs
        gamma.filters = [gamma_only]
        # A Compton step can transfer energy to an electron with zero local edep.
        gamma.keep_zero_edep = True
        gamma.output_filename = str(output_dir / f"gamma_steps_{label}.root")

        edep = sim.add_actor("DigitizerHitsCollectionActor", f"lyso_edep_{label}")
        edep.attached_to = volumes["pixels"]
        edep.filters = [nonoptical]
        edep.attributes = ids + ["ParticleName", "TotalEnergyDeposit", "PostPosition", "TrackVolumeName", "GlobalTime"]
        edep.output_filename = str(output_dir / f"lyso_edep_{label}.root")

        if optical:
            sipm = sim.add_actor("PhaseSpaceActor", f"sipm_photons_{label}")
            sipm.attached_to = volumes["sipm"]
            sipm.steps_to_store = "entering"
            sipm.filters = [optical_only]
            sipm.attributes = ids + ["PrePosition", "PrePositionLocal", "KineticEnergy", "GlobalTime"]
            sipm.output_filename = str(output_dir / f"sipm_hits_{label}.root")
            kill = sim.add_actor("KillActor", f"kill_late_optical_{label}")
            kill.attached_to = volumes["pixels"]
            kill.filters = [optical_only, late]
            kill.filters_boolean_operator = "and"

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = str(output_dir / "stats.json")
    return stats


def build_simulation(args):
    import opengate as gate
    from opengate.utility import g4_units as u

    sim = gate.Simulation()
    sim.number_of_threads = args.threads
    sim.random_seed = args.seed
    sim.progress_bar = True
    sim.check_volumes_overlap = args.check_overlaps
    sim.visu = args.visu
    sim.visu_type = "qt"
    # Full apparatus must fit: A is beyond +300 mm, B beyond -300 mm.
    sim.world.size = [1000 * u.mm, 1000 * u.mm, (args.gap_mm + 400) * u.mm]
    sim.world.material = "G4_AIR"
    # Hide the enclosing air box, otherwise solid rendering obscures the apparatus.
    sim.world.color = [1.0, 1.0, 1.0, 0.0]
    pm = sim.physics_manager
    pm.physics_list_name = "G4EmStandardPhysics"  # Effective list in the original script
    pm.special_physics_constructors.G4OpticalPhysics = not args.no_optical
    for particle in ("gamma", "electron", "positron"):
        pm.set_production_cut("world", particle, 0.1 * u.mm)
    pm.optical_properties_file = str(HERE / "optical_properties_fast.xml")
    pm.surface_properties_file = str(HERE / "surface_properties_fast.xml")

    # Inherited composition, including its original LSO-like LYSO approximation.
    sim.volume_manager.material_database.add_material_nb_atoms(
        "LYSO", ["Lu", "Si", "O"], [2, 1, 5], 7.1 * u.g / u.cm**3
    )
    layout = module_layout(args.gap_mm)
    modules = {label: add_module(sim, label, placement, u, not args.no_optical)
               for label, placement in layout.items()}

    # Provisional plastic disk, not a manufacturer's exact source construction.
    capsule = sim.add_volume("Tubs", "source_capsule")
    capsule.mother = "world"
    capsule.rmin = 0
    capsule.rmax = args.capsule_radius_mm * u.mm
    capsule.dz = args.capsule_thickness_mm * u.mm / 2
    capsule.sphi = 0
    capsule.dphi = 360 * u.deg
    capsule.material = "G4_PLEXIGLASS"
    capsule.color = [1.0, 0.8, 0.1, 0.7]

    source_metadata = add_source(sim, args, layout)

    stats = add_scoring(sim, modules, capsule.name, args.output_dir, not args.no_optical, u)
    if args.visu:
        commands = (HERE / "lab-view.mac").read_text().splitlines()
        sim.visu_commands = [s.strip() for s in commands if s.strip() and not s.lstrip().startswith("#")]
    return sim, stats, layout, source_metadata


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim, stats, layout, source_metadata = build_simulation(args)
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "opengate_version": version("opengate"),
        "expected_source_events": args.events * args.threads,
        "source_position_mm": [0, 0, 0],
        "capsule_material": "G4_PLEXIGLASS (provisional)",
        "front_glass_thickness_mm": GLASS_MM,
        "pixel_size_mm": PIXEL_MM.tolist(),
        "pitch_mm": PITCH_MM,
        "modules": {label: {k: v.tolist() for k, v in placement.items()} for label, placement in layout.items()},
        "polarization_entanglement_modelled": False,
        "status": "started",
    }
    metadata.update(source_metadata)
    manifest = args.output_dir / "run_config.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print("Each sampled 511-keV pair crosses both detector-array apertures.", flush=True)
    sim.run()
    print(stats)
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
