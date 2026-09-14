from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.report_phase2_decoders import paired_contract_interval


class ReportingTests(unittest.TestCase):
    def test_contract_paired_interval_checks_identity_and_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); common={"targets":np.array([0.,1.,2.,3.],np.float32),"row_indices":np.arange(4),
                "contract_ids":np.array([0,0,1,1],np.int32),"window_starts":np.arange(4)}
            np.savez(root/"d0.npz",predictions=np.array([1.,2.,3.,4.],np.float32),**common)
            np.savez(root/"d1.npz",predictions=np.array([.5,1.5,2.5,3.5],np.float32),**common)
            result=paired_contract_interval(root/"d1.npz",root/"d0.npz","mae",replicates=20,seed=1)
            self.assertAlmostEqual(result["difference"],-0.5)
            common["row_indices"]=np.array([0,1,2,9])
            np.savez(root/"bad.npz",predictions=np.zeros(4,np.float32),**common)
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                paired_contract_interval(root/"bad.npz",root/"d0.npz","mae",replicates=2)


if __name__ == "__main__": unittest.main()
