from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.phase2_decoders.runner import DecoderRunConfig, run_decoder_experiment


class DecoderRunnerTests(unittest.TestCase):
    def test_one_epoch_cpu_run_writes_replayable_artifacts(self) -> None:
        rng=np.random.default_rng(4); dims={"statistical":2,"transformed":2,"vae":2,"contrastive":2,"byol":2}
        train=rng.normal(size=(12,10)).astype(np.float32); test=rng.normal(size=(8,10)).astype(np.float32)
        train_contexts=np.stack([np.arange(i-2,i+1) for i in range(2,11)])
        test_contexts=np.stack([np.arange(i-2,i+1) for i in range(2,8)])
        y_train=rng.normal(size=len(train_contexts)).astype(np.float32); y_test=rng.normal(size=len(test_contexts)).astype(np.float32)
        identities={}
        for split,contexts in (("train",train_contexts),("test",test_contexts)):
            n=len(contexts); identities[f"{split}_row_indices"]=contexts[:,-1]; identities[f"{split}_contract_ids"]=np.zeros(n,np.int32)
            identities[f"{split}_window_starts"]=contexts[:,-1]; identities[f"{split}_timestamps_ns"]=contexts[:,-1]
        scaler={f"{name}__mean":np.zeros(width,np.float32) for name,width in dims.items()}
        scaler.update({f"{name}__std":np.ones(width,np.float32) for name,width in dims.items()})
        with tempfile.TemporaryDirectory() as tmp:
            run_decoder_experiment(train_features=train,test_features=test,train_contexts=train_contexts,test_contexts=test_contexts,
                y_train=y_train,y_test=y_test,identities=identities,branch_dims=dims,scaler=scaler,run_root=tmp,
                config=DecoderRunConfig("price_prediction","D0",3,(1,),0,4,1e-4,0.0,"cpu"),dataset_manifest={})
            root=Path(tmp); self.assertTrue((root/"e1/checkpoint.pth").exists())
            replay=json.loads((root/"e1/replay.json").read_text())
            self.assertTrue(replay["verified"])
            self.assertTrue(replay["prediction_values_verified"])
            self.assertIn("device_name",json.loads((root/"timing.json").read_text()))
            with np.load(root/"e1/predictions.npz") as saved: np.testing.assert_array_equal(saved["row_indices"],test_contexts[:,-1])


if __name__ == "__main__": unittest.main()
