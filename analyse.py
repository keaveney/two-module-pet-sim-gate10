"""Reduce one run to an event-wise student CSV; all lengths mm, energies keV, times ns."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detector_response import ROOT_KEYS, read_root, reduce_module


def angle_degrees(a, b):
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide((a * b).sum(axis=1), denominator,
                       out=np.full(len(a), np.nan), where=denominator > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def module_truth(folder, module, config):
    """First two physical LYSO steps of the earliest primary 511 track in this module.

    Rayleigh steps are retained: 'second' always means the next physical step,
    never the next conveniently selected Compton/photoelectric step.
    """
    vectors = ['PostPosition', 'PreDirection', 'PostDirection']
    branches = ROOT_KEYS + ['TrackID', 'ParentID', 'TrackVertexKineticEnergy',
                           'ProcessDefinedStep', 'GlobalTime', 'PreKineticEnergy',
                           'PostKineticEnergy', 'TotalEnergyDeposit', 'TrackVolumeName']
    branches += [f'{v}_{axis}' for v in vectors for axis in 'XYZ']
    steps = read_root(folder / f'gamma_steps_{module}.root', branches)
    process = steps['ProcessDefinedStep'].fillna('').str.lower()
    # Vertex energy identifies the original photon even after capsule scattering.
    keep = ((steps['ParentID'] == 0)
            & np.isclose(steps['TrackVertexKineticEnergy'], 0.51099895, atol=1e-5, rtol=0)
            & ~process.isin(['', 'transportation', 'coupledtransportation', 'userspecialcut', 'steplimiter']))
    steps = steps.loc[keep].sort_values(ROOT_KEYS + ['GlobalTime', 'TrackID']).copy()
    result = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=ROOT_KEYS))
    result['truth_n_primary_tracks'] = steps.groupby(ROOT_KEYS)['TrackID'].nunique()
    selected = steps.drop_duplicates(ROOT_KEYS)[ROOT_KEYS + ['TrackID']]
    steps = steps.merge(selected, on=ROOT_KEYS + ['TrackID'], how='inner')
    steps = steps.sort_values(ROOT_KEYS + ['GlobalTime'])
    steps['interaction_number'] = steps.groupby(ROOT_KEYS).cumcount() + 1
    result['truth_n_interactions'] = steps.groupby(ROOT_KEYS).size()
    placement = config['modules'][module]
    centre, rotation = np.asarray(placement['centre_mm']), np.asarray(placement['rotation'])
    for number in (1, 2):
        hit = steps.loc[steps['interaction_number'] == number].set_index(ROOT_KEYS)
        prefix = f'truth_i{number}_'
        result[prefix + 'track_id'] = hit['TrackID']
        result[prefix + 'process'] = hit['ProcessDefinedStep']
        result[prefix + 'type'] = hit['ProcessDefinedStep'].str.lower().map(
            {'phot': 'PE', 'compt': 'CS', 'rayl': 'Rayleigh'}).fillna('other')
        result[prefix + 'pixel'] = hit['TrackVolumeName']
        result[prefix + 'time_ns'] = hit['GlobalTime']
        for column, branch in [('energy_before_kev', 'PreKineticEnergy'),
                               ('energy_after_kev', 'PostKineticEnergy'),
                               ('local_edep_kev', 'TotalEnergyDeposit')]:
            result[prefix + column] = hit[branch] * 1000
        result[prefix + 'energy_transfer_kev'] = (hit['PreKineticEnergy'] - hit['PostKineticEnergy']) * 1000
        world = hit[[f'PostPosition_{a}' for a in 'XYZ']].to_numpy(float)
        local = (world - centre) @ rotation
        for axis, k in zip('xyz', range(3)):
            result[prefix + f'world_{axis}_mm'] = pd.Series(world[:, k], index=hit.index)
            result[prefix + f'local_{axis}_mm'] = pd.Series(local[:, k], index=hit.index)
            for direction in ['PreDirection', 'PostDirection']:
                label = 'incoming' if direction == 'PreDirection' else 'outgoing'
                result[prefix + f'{label}_{axis}'] = hit[f'{direction}_{axis.upper()}']
        result[prefix + 'depth_mm'] = pd.Series(local[:, 2] + config['pixel_size_mm'][2] / 2, index=hit.index)

    def values(prefix):
        return result[[prefix + a for a in 'xyz']].to_numpy(float)

    incoming = values('truth_i1_incoming_')
    outgoing = values('truth_i1_outgoing_')
    world1 = result[[f'truth_i1_world_{a}_mm' for a in 'xyz']].to_numpy(float)
    world2 = result[[f'truth_i2_world_{a}_mm' for a in 'xyz']].to_numpy(float)
    displacement = world2 - world1
    is_cs = result['truth_i1_type'].eq('CS').to_numpy()
    result['truth_compton_angle_deg'] = np.where(is_cs, angle_degrees(incoming, outgoing), np.nan)
    result['truth_two_site_angle_deg'] = np.where(is_cs, angle_degrees(incoming, displacement), np.nan)
    result['truth_site_separation_mm'] = np.linalg.norm(displacement, axis=1)
    before, after = result['truth_i1_energy_before_kev'], result['truth_i1_energy_after_kev']
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine = 1 - 510.99895 * (1 / after - 1 / before)
    valid = is_cs & (after > 0) & cosine.between(-1, 1)
    result['truth_energy_angle_deg'] = np.where(valid, np.degrees(np.arccos(np.clip(cosine, -1, 1))), np.nan)
    return result.reset_index()


def analyse(folder, pde=0.30, seed=12345):
    config = json.loads((folder / 'run_config.json').read_text())
    if config['status'] != 'complete':
        raise ValueError('Analyse only completed runs.')
    # GATE can offset EventIDs by worker, so read the generated primary ledger.
    primaries = read_root(folder / 'source_primaries.root', ROOT_KEYS + ['ParentID', 'ThreadID'])
    frame = (primaries.loc[primaries.ParentID.eq(0), ROOT_KEYS + ['ThreadID']]
             .drop_duplicates().sort_values(ROOT_KEYS).reset_index(drop=True))
    if len(frame) != config['expected_source_events'] or frame.duplicated(ROOT_KEYS).any():
        raise ValueError('Primary ledger is incomplete or event identifiers are not unique.')
    for offset, module in enumerate(('A', 'B')):
        truth = module_truth(folder, module, config)
        optical_path = folder / f'sipm_hits_{module}.root'
        if not config['arguments']['no_optical'] and not optical_path.exists():
            raise FileNotFoundError(optical_path)
        response = (reduce_module(folder, module, pde, seed + offset)
                    if optical_path.exists() else pd.DataFrame(columns=ROOT_KEYS))
        for table in (truth, response):
            if not table.empty and not pd.MultiIndex.from_frame(table[ROOT_KEYS]).isin(pd.MultiIndex.from_frame(frame[ROOT_KEYS])).all():
                raise ValueError('Unexpected event IDs; refusing to drop unrecognised events.')
            frame = frame.merge(table.rename(columns={c: module + '_' + c for c in table if c not in ROOT_KEYS}),
                                on=ROOT_KEYS, how='left', validate='one_to_one')
        channels = [f'{module}_npe_{x}_{y}' for x in range(8) for y in range(8)]
        count_columns = channels + [f'{module}_npe_total', f'{module}_max_npe',
                                    f'{module}_n_optical_total', f'{module}_max_n_optical']
        missing = {column: 0 for column in count_columns if column not in frame}
        missing.update({f'{module}_{column}': np.nan for column in ['first_pe_time_ns', 'max_ix', 'max_iy']
                        if f'{module}_{column}' not in frame})
        frame = pd.concat([frame, pd.DataFrame(missing, index=frame.index)], axis=1)
        frame[count_columns] = frame[count_columns].fillna(0).astype(int)
        frame = frame.copy()
        total = frame[f'{module}_npe_total']
        frame[f'{module}_detected'] = total > 0
        frame[f'{module}_w'] = frame[f'{module}_max_npe'] / total.replace(0, np.nan)
        for column in ['truth_n_primary_tracks', 'truth_n_interactions']:
            frame[f'{module}_{column}'] = frame[f'{module}_{column}'].fillna(0).astype(int)
        edep = read_root(folder / f'lyso_edep_{module}.root', ROOT_KEYS + ['TotalEnergyDeposit'])
        total_edep = edep.groupby(ROOT_KEYS)['TotalEnergyDeposit'].sum().mul(1000).rename(f'{module}_truth_total_edep_kev')
        frame = frame.merge(total_edep, on=ROOT_KEYS, how='left')
        frame[f'{module}_truth_total_edep_kev'] = frame[f'{module}_truth_total_edep_kev'].fillna(0)

    frame['both_detected'] = frame.A_detected & frame.B_detected
    frame['truth_distinct_pair'] = (frame.A_truth_n_primary_tracks.eq(1) & frame.B_truth_n_primary_tracks.eq(1)
                                  & frame.A_truth_i1_track_id.ne(frame.B_truth_i1_track_id))
    normals = []
    for m in ('A', 'B'):
        incoming = frame[[f'{m}_truth_i1_incoming_{a}' for a in 'xyz']].to_numpy(float)
        normals.append(np.cross(incoming, frame[[f'{m}_truth_i1_outgoing_{a}' for a in 'xyz']].to_numpy(float)))
    plane_angle = angle_degrees(*normals)
    # Scattering planes are unoriented: fold their separation into [0, 90] degrees.
    frame['truth_plane_angle_deg'] = np.where(
        frame.truth_distinct_pair & frame.A_truth_i1_type.eq('CS') & frame.B_truth_i1_type.eq('CS'),
        np.minimum(plane_angle, 180 - plane_angle), np.nan)
    frame.insert(0, 'simulation_id', folder.resolve().name)
    frame['pixel_length_mm'] = config['pixel_size_mm'][2]
    frame['pde'] = pde
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--pde', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    if not 0 < args.pde <= 1:
        parser.error('PDE must be in (0, 1].')
    frame = analyse(args.input_dir, args.pde, args.seed)
    output = args.input_dir / 'events.csv'
    frame.to_csv(output, index=False)
    print(f'Wrote {len(frame)} events to {output}; {frame.both_detected.sum()} detected in both modules.')


if __name__ == '__main__':
    main()
