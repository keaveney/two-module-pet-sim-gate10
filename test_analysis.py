"""Small checks of interaction ordering and coordinate/angle conventions."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyse import module_truth


class TruthTests(unittest.TestCase):
    def test_track_order_coordinates_and_missing_second_site(self):
        def hit(event, track, time, process, position, before, after, parent=0, vertex=0.51099895):
            row = dict(RunID=0, EventID=event, TrackID=track, ParentID=parent,
                       TrackVertexKineticEnergy=vertex, GlobalTime=time,
                       ProcessDefinedStep=process, PreKineticEnergy=before,
                       PostKineticEnergy=after, TotalEnergyDeposit=0,
                       TrackVolumeName='B_pixel_1_2')
            for key, vector in [('PostPosition', position), ('PreDirection', [0, 0, -1]),
                                ('PostDirection', [1, 0, 0])]:
                row.update({f'{key}_{axis}': value for axis, value in zip('XYZ', vector)})
            return row

        steps = pd.DataFrame([
            hit(0, 2, 1, 'Transportation', [0, 0, -300], .511, .511),
            hit(0, 2, 2, 'compt', [0, 0, -310], .51099895, .255499475),
            hit(0, 2, 3, 'phot', [1, 0, -310], .255499475, 0),
            hit(0, 3, 2.5, 'phot', [2, 0, -310], .511, 0),
            hit(1, 2, 1, 'phot', [0, 0, -310], .511, 0),
            hit(2, 2, 1, 'Rayl', [0, 0, -310], .511, .511),
            hit(2, 2, 2, 'compt', [1, 0, -310], .511, .3),
            hit(3, 1, 1, 'compt', [0, 0, -310], 1.2745, .6, vertex=1.2745),
            hit(3, 5, 2, 'phot', [1, 0, -310], .511, 0, parent=1),
        ])
        config = {'pixel_size_mm': [3, 3, 20], 'modules': {
            'B': {'centre_mm': [0, 0, -311], 'rotation': np.diag([-1, 1, -1]).tolist()}}}
        with patch('analyse.read_root', return_value=steps):
            truth = module_truth(Path('.'), 'B', config).set_index('EventID')
        self.assertEqual(truth.loc[0, 'truth_n_primary_tracks'], 2)
        self.assertEqual(truth.loc[0, 'truth_i2_track_id'], 2)
        self.assertEqual(truth.loc[0, 'truth_i1_type'], 'CS')
        self.assertEqual(truth.loc[0, 'truth_i2_type'], 'PE')
        self.assertEqual(truth.loc[0, 'truth_i1_depth_mm'], 9)
        self.assertEqual(truth.loc[0, 'truth_i2_local_x_mm'], -1)
        self.assertAlmostEqual(truth.loc[0, 'truth_two_site_angle_deg'], 90)
        self.assertAlmostEqual(truth.loc[0, 'truth_energy_angle_deg'], 90)
        self.assertTrue(pd.isna(truth.loc[1, 'truth_i2_type']))
        self.assertEqual(truth.loc[2, 'truth_i1_type'], 'Rayleigh')
        self.assertNotIn(3, truth.index)
        with patch('analyse.read_root', return_value=steps.iloc[:0]):
            empty = module_truth(Path('.'), 'B', config)
        self.assertTrue(empty.empty)
        self.assertIn('truth_i2_depth_mm', empty)


if __name__ == '__main__':
    unittest.main()
