# =========================================================================
# Copyright (C) 2024. Cloud-Device Recommendation System.
# =========================================================================

"""
Base Stage Abstract Class

This module defines the abstract base class for all pipeline stages.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any
import logging
import os
from ..config.feature_groups import FeatureGroupManager, FeatureGroup


class StageType(Enum):
    """Pipeline stage types"""
    RETRIEVAL = "retrieval"
    PRERANKING = "preranking"
    RERANKING = "reranking"


class BaseStage(ABC):
    """
    Abstract base class for pipeline stages.

    All stages (Retrieval, Pre-ranking, Re-ranking) inherit from this class.
    """

    def __init__(self,
                 stage_name: str,
                 stage_type: StageType,
                 feature_group_manager: FeatureGroupManager,
                 allowed_feature_groups: List[FeatureGroup],
                 output_dir: str = "./outputs",
                 **kwargs):
        """
        Initialize base stage.

        Args:
            stage_name: Unique name for this stage instance
            stage_type: Type of stage (retrieval, preranking, reranking)
            feature_group_manager: Manager for feature group filtering
            allowed_feature_groups: Which feature groups this stage can use
            output_dir: Directory for saving stage outputs
        """
        self.stage_name = stage_name
        self.stage_type = stage_type
        self.feature_group_manager = feature_group_manager
        self.allowed_feature_groups = allowed_feature_groups
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{stage_name}]")
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate no FG3 in cloud stages
        # if stage_type in [StageType.RETRIEVAL, StageType.PRERANKING]:
        #     if FeatureGroup.FG3 in allowed_feature_groups:
        #         raise ValueError(f"Cloud stage '{stage_name}' cannot use FG3 features!")

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Initialized {stage_type.value} stage: {stage_name}")
        self.logger.info(f"Allowed feature groups: {[fg.value for fg in allowed_feature_groups]}")

    @abstractmethod
    def train(self,
              train_data: Any,
              valid_data: Optional[Any] = None,
              **kwargs) -> Dict[str, float]:
        """
        Train the stage's model.

        Args:
            train_data: Training data
            valid_data: Validation data
            **kwargs: Training parameters

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def evaluate(self,
                 test_data: Any,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate the stage on test data.

        Args:
            test_data: Test data
            **kwargs: Evaluation parameters

        Returns:
            Dictionary of evaluation metrics
        """
        pass
