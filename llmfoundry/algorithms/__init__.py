# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.algorithms import (
    Alibi,
    GatedLinearUnits,
    GradientClipping,
    LowPrecisionLayerNorm,
)

from llmfoundry.registry import algorithms

algorithms.register('gradient_clipping', func=GradientClipping)
algorithms.register('alibi', func=Alibi)
algorithms.register('gated_linear_units', func=GatedLinearUnits)
algorithms.register('low_precision_layernorm', func=LowPrecisionLayerNorm)

# ===== Start of Mask Pruned Weights Algorithm =====
from composer.core import Algorithm, Event
import torch
class MaskPrunedWeights(Algorithm):
    def match(self, event, state):
        # masking weights after optimizer step should be sufficient
        # if we detect weird behaviour, we can also do it before the forward pass
        # by adding `or event == Event.BATCH_START`
        return event == Event.BATCH_END

    @torch.no_grad()
    def apply(self, event, state, logger):
        def mask_weights(module):
            if hasattr(module, 'mask'):
                module.weight *= module.mask

        state.model.apply(mask_weights)

algorithms.register('mask_pruned_weights', func=MaskPrunedWeights)
# ===== End of Mask Pruned Weights Algorithm =====
