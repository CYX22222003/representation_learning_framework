from __future__ import annotations

import unittest

import numpy as np

from tasks.phase2_decoders.data import (
    align_contexts_to_labels,
    build_price_label_bundle,
    build_temporal_index,
    validate_regression_labels,
    validate_temporal_index,
)


class TemporalDataTests(unittest.TestCase):
    def test_contexts_remain_inside_contract_splits(self) -> None:
        times=[np.arange(9,dtype=np.int64),np.arange(20,27,dtype=np.int64)]
        bundle=build_temporal_index([(5,4),(4,3)],context_length=3,contract_timestamps_ns=times)
        result=validate_temporal_index(bundle,train_size=9,test_size=7)
        self.assertEqual(result["train_context_count"],5)
        self.assertEqual(result["test_context_count"],3)
        np.testing.assert_array_equal(bundle["train_context_row_indices"][2],np.array([2,3,4]))
        np.testing.assert_array_equal(bundle["train_context_row_indices"][3],np.array([5,6,7]))
        self.assertEqual(bundle["train_contract_ids"].tolist(),[0,0,0,1,1])

    def test_validation_rejects_noncontiguous_context(self) -> None:
        bundle=build_temporal_index([(5,4)],context_length=3)
        bundle["train_context_row_indices"][0,1]=4
        with self.assertRaisesRegex(ValueError,"non-contiguous"):
            validate_temporal_index(bundle,train_size=5,test_size=4)

    def test_price_labels_and_temporal_join_are_contract_safe(self) -> None:
        contracts=[]
        for base,count in ((0.0,9),(100.0,7)):
            seq=np.zeros((count,2,5),dtype=np.float32)
            seq[:,-1,3]=base+np.arange(count)
            contracts.append(seq)
        labels=build_price_label_bundle(contracts,horizon=1,train_ratio=0.6)
        validate_regression_labels(labels,train_size=9,test_size=7)
        temporal=build_temporal_index([(5,4),(4,3)],context_length=3)
        contexts,target,ids=align_contexts_to_labels(temporal,labels,"train")
        self.assertEqual(len(target),3)
        self.assertTrue(np.all(np.diff(contexts,axis=1)==1))
        self.assertEqual(ids["contract_ids"].tolist(),[0,0,1])
        self.assertLess(float(target[:2].max()),100.0)


if __name__ == "__main__": unittest.main()
