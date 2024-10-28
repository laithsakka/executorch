# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List

from executorch.exir.scalar_type import ScalarType

# Note: check executorch/schema/data.fbs for explanations of these fields.


@dataclass
class TensorMetadata:
    fully_qualified_name: str
    scalar_type: ScalarType
    dim_sizes: List[int]
    dim_order: List[bytes]
    storage_offset: int
    layout: int

    offset: int
    size: int


@dataclass
class TensorSegment:
    segment_index: int
    tensor_metadata: List[TensorMetadata]


@dataclass
class DataSegment:
    offset: int
    size: int


@dataclass
class Data:
    version: int
    tensor_alignment: int
    tensor_segments: List[TensorSegment]
    data_segments: List[DataSegment]
