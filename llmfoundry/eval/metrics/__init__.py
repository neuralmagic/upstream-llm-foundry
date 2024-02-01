# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from llmfoundry.eval.metrics.nlp import (
    InContextLearningGenerationExactMatchAccuracy,
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMetric,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningPerplexity,
    InContextLearningCrossEntropy,
)

__all__ = [
    'InContextLearningMetric',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationExactMatchAccuracy',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningPerplexity',
    'InContextLearningCrossEntropy',
]
