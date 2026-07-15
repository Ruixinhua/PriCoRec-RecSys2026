# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Disabled DP-SGD model placeholder.

The previous implementation clipped one aggregate batch gradient and then
reported an RDP epsilon. Aggregate clipping is not per-sample DP-SGD, so that
epsilon was not a valid privacy guarantee. This class now fails closed until a
maintained implementation with verified per-sample clipping and accounting is
integrated.
"""

from fuxictr.pytorch.models import BaseModel


class DPSGD(BaseModel):
    """Prevent use of the unverified historical DP-SGD implementation."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DPSGD is disabled because PriCoRec does not currently provide a "
            "verified per-sample DP-SGD implementation or privacy accountant. "
            "Use a maintained differential-privacy library and validate its "
            "sampling, clipping, noise, and accounting configuration before "
            "making any (epsilon, delta)-DP claim."
        )
