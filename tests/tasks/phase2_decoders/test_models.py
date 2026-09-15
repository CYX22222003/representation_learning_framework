from __future__ import annotations

import unittest

import torch

from tasks.phase2_decoders.models import make_decoder


DIMS={"statistical":70,"transformed":55,"vae":64,"contrastive":128,"byol":128}


class DecoderModelTests(unittest.TestCase):
    def test_shapes_and_frozen_parameter_counts(self) -> None:
        expected={"D0":65409,"D1":211713,"D2":165126,"D3":198017,"D4":197473}
        for decoder_id,count in expected.items():
            model=make_decoder(decoder_id,DIMS,"price_prediction",8)
            x=torch.zeros(2,8,445) if decoder_id in {"D3","D4"} else torch.zeros(2,445)
            self.assertEqual(tuple(model(x).shape),(2,1))
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
        matched=[expected[key] for key in ("D1","D3","D4")]
        self.assertLess(max(matched)/min(matched),1.10)

    def test_volatility_is_nonnegative_and_gates_are_probabilities(self) -> None:
        for decoder_id in ("D0","D1","D2","D3","D4"):
            model=make_decoder(decoder_id,DIMS,"volatility_prediction",8).eval()
            x=torch.randn(3,8,445) if decoder_id in {"D3","D4"} else torch.randn(3,445)
            output,aux=model.forward_with_aux(x)
            self.assertTrue(torch.all(output>=0))
            if decoder_id=="D2":
                self.assertEqual(tuple(aux.shape),(3,5))
                torch.testing.assert_close(aux.sum(1),torch.ones(3))

    def test_transformer_is_deterministic_in_eval_mode(self) -> None:
        model=make_decoder("D4",DIMS,"price_prediction",8).eval(); x=torch.randn(2,8,445)
        torch.testing.assert_close(model(x),model(x))


if __name__ == "__main__": unittest.main()
