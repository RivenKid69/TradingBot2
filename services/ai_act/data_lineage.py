# -*- coding: utf-8 -*-
"""
AI Act Data Lineage Tracking System.

This module provides data provenance tracking for EU AI Act compliance,
supporting Article 10 (Data Governance) and Article 11 (Technical Documentation).

Data lineage tracking is essential for:
1. Traceability - Understanding data origin and transformations
2. Reproducibility - Recreating datasets from source to model
3. Audit - Demonstrating compliance with data governance requirements
4. Impact Analysis - Understanding effects of data changes on model

This module provides:
1. DataLineageTracker - Main lineage tracking system
2. DataNode - Represents a data artifact in the lineage graph
3. DataTransformation - Records transformations applied to data
4. LineageGraph - Graph representation of data flow

References:
    - Article 10: https://artificialintelligenceact.eu/article/10/
    - Article 11: https://artificialintelligenceact.eu/article/11/
    - Annex IV: https://artificialintelligenceact.eu/annex/4/
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class DataNodeType(Enum):
    """Types of data nodes in the lineage graph."""
    RAW_SOURCE = "raw_source"              # Original data source
    EXTRACTED = "extracted"                 # Data extracted from source
    CLEANED = "cleaned"                     # Cleaned/preprocessed data
    TRANSFORMED = "transformed"             # Transformed features
    AGGREGATED = "aggregated"               # Aggregated data
    NORMALIZED = "normalized"               # Normalized/scaled data
    FILTERED = "filtered"                   # Filtered subset
    MERGED = "merged"                       # Merged from multiple sources
    SPLIT = "split"                         # Split (train/val/test)
    ANNOTATED = "annotated"                 # Annotated/labeled data
    MODEL_INPUT = "model_input"             # Final input to model
    MODEL_OUTPUT = "model_output"           # Model predictions
    CACHED = "cached"                       # Cached intermediate data


class TransformationType(Enum):
    """Types of data transformations."""
    # Cleaning
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_REMOVAL = "outlier_removal"
    OUTLIER_WINSORIZATION = "outlier_winsorization"
    DUPLICATE_REMOVAL = "duplicate_removal"
    DATA_TYPE_CONVERSION = "data_type_conversion"

    # Transformation
    FEATURE_ENGINEERING = "feature_engineering"
    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
    LOG_TRANSFORM = "log_transform"
    DIFFERENCING = "differencing"
    LAGGING = "lagging"
    ROLLING_WINDOW = "rolling_window"

    # Aggregation
    RESAMPLING = "resampling"
    GROUPBY_AGGREGATION = "groupby_aggregation"
    PIVOT = "pivot"

    # Filtering
    TIME_FILTER = "time_filter"
    VALUE_FILTER = "value_filter"
    COLUMN_SELECTION = "column_selection"
    ROW_SAMPLING = "row_sampling"

    # Joining
    MERGE = "merge"
    CONCATENATION = "concatenation"
    JOIN = "join"

    # Splitting
    TRAIN_TEST_SPLIT = "train_test_split"
    WALK_FORWARD_SPLIT = "walk_forward_split"
    STRATIFIED_SPLIT = "stratified_split"

    # Shifting (critical for trading - prevents look-ahead bias)
    FEATURE_SHIFT = "feature_shift"
    TARGET_ALIGNMENT = "target_alignment"

    # Custom
    CUSTOM = "custom"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DataNode:
    """
    Represents a data artifact in the lineage graph.

    Each node captures:
    - Identity: Unique ID and name
    - Type: Classification of the data artifact
    - Provenance: Source information
    - Properties: Schema, statistics, quality metrics
    - Metadata: Version, timestamps, tags
    """
    node_id: str = ""
    name: str = ""
    node_type: str = ""
    description: str = ""

    # Provenance
    source_system: str = ""       # E.g., "binance", "alpaca", "internal"
    source_location: str = ""     # E.g., file path, API endpoint, table name
    extraction_method: str = ""   # E.g., "api_call", "file_read", "database_query"

    # Schema
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    record_count: int = 0

    # Statistics (subset - full stats in DatasetMetadata)
    statistics_hash: str = ""  # Hash of full statistics for change detection

    # Quality
    quality_score: float = 0.0
    quality_check_id: str = ""

    # Relationships
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    transformation_ids: List[str] = field(default_factory=list)

    # Versioning
    version: str = "1.0.0"
    content_hash: str = ""  # Hash of actual data content

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Tags for classification
    tags: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"DN-{uuid.uuid4().hex[:12].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataNode":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DataTransformation:
    """
    Records a data transformation in the lineage.

    Captures:
    - What transformation was applied
    - Input and output nodes
    - Parameters used
    - Code/function reference
    - Validation status
    """
    transformation_id: str = ""
    transformation_type: str = ""
    name: str = ""
    description: str = ""

    # Input/Output
    input_node_ids: List[str] = field(default_factory=list)
    output_node_ids: List[str] = field(default_factory=list)

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Code reference (Article 10(2)(b) - preparation processes)
    function_name: str = ""
    module_name: str = ""
    code_version: str = ""
    code_hash: str = ""

    # Validation
    validated: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # Execution details
    execution_time_seconds: float = 0.0
    executed_by: str = ""
    execution_environment: str = ""

    # Timestamps
    executed_at: str = ""

    # Tags
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.transformation_id:
            self.transformation_id = f"DT-{uuid.uuid4().hex[:12].upper()}"
        if not self.executed_at:
            self.executed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataTransformation":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LineageEdge:
    """An edge in the lineage graph connecting nodes through transformations."""
    edge_id: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    transformation_id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = f"LE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataLineageConfig:
    """Configuration for Data Lineage Tracking."""
    # Storage
    storage_path: str = "logs/ai_act/data_lineage"
    enable_persistence: bool = True

    # Hashing
    enable_content_hashing: bool = True
    hash_sample_size: int = 10000  # Sample size for large datasets

    # Validation
    enable_transformation_validation: bool = True
    validate_row_count_changes: bool = True
    validate_column_changes: bool = True

    # Graph
    max_lineage_depth: int = 100  # Maximum depth of lineage traversal

    # Logging
    log_all_transformations: bool = True


# =============================================================================
# Lineage Graph
# =============================================================================


class LineageGraph:
    """
    Graph representation of data lineage.

    Provides traversal and querying capabilities for the lineage graph.
    """

    def __init__(self):
        self._nodes: Dict[str, DataNode] = {}
        self._transformations: Dict[str, DataTransformation] = {}
        self._edges: Dict[str, LineageEdge] = {}
        self._lock = threading.RLock()

    def add_node(self, node: DataNode) -> None:
        """Add a node to the graph."""
        with self._lock:
            self._nodes[node.node_id] = node

    def add_transformation(self, transformation: DataTransformation) -> None:
        """Add a transformation to the graph."""
        with self._lock:
            self._transformations[transformation.transformation_id] = transformation

            # Create edges
            for input_id in transformation.input_node_ids:
                for output_id in transformation.output_node_ids:
                    edge = LineageEdge(
                        source_node_id=input_id,
                        target_node_id=output_id,
                        transformation_id=transformation.transformation_id,
                    )
                    self._edges[edge.edge_id] = edge

                    # Update node relationships
                    if input_id in self._nodes:
                        if output_id not in self._nodes[input_id].child_ids:
                            self._nodes[input_id].child_ids.append(output_id)
                        if transformation.transformation_id not in self._nodes[input_id].transformation_ids:
                            self._nodes[input_id].transformation_ids.append(transformation.transformation_id)

                    if output_id in self._nodes:
                        if input_id not in self._nodes[output_id].parent_ids:
                            self._nodes[output_id].parent_ids.append(input_id)
                        if transformation.transformation_id not in self._nodes[output_id].transformation_ids:
                            self._nodes[output_id].transformation_ids.append(transformation.transformation_id)

    def get_node(self, node_id: str) -> Optional[DataNode]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_transformation(self, transformation_id: str) -> Optional[DataTransformation]:
        """Get a transformation by ID."""
        with self._lock:
            return self._transformations.get(transformation_id)

    def get_ancestors(self, node_id: str, max_depth: int = 100) -> List[DataNode]:
        """Get all ancestor nodes (upstream lineage)."""
        ancestors: List[DataNode] = []
        visited: Set[str] = set()

        def _traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            node = self._nodes.get(current_id)
            if not node:
                return

            for parent_id in node.parent_ids:
                if parent_id not in visited:
                    parent = self._nodes.get(parent_id)
                    if parent:
                        ancestors.append(parent)
                    _traverse(parent_id, depth + 1)

        with self._lock:
            _traverse(node_id, 0)

        return ancestors

    def get_descendants(self, node_id: str, max_depth: int = 100) -> List[DataNode]:
        """Get all descendant nodes (downstream lineage)."""
        descendants: List[DataNode] = []
        visited: Set[str] = set()

        def _traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            node = self._nodes.get(current_id)
            if not node:
                return

            for child_id in node.child_ids:
                if child_id not in visited:
                    child = self._nodes.get(child_id)
                    if child:
                        descendants.append(child)
                    _traverse(child_id, depth + 1)

        with self._lock:
            _traverse(node_id, 0)

        return descendants

    def get_transformation_chain(self, node_id: str) -> List[DataTransformation]:
        """Get the chain of transformations that produced a node."""
        transformations: List[DataTransformation] = []
        visited: Set[str] = set()

        def _traverse(current_id: str):
            node = self._nodes.get(current_id)
            if not node:
                return

            for trans_id in node.transformation_ids:
                if trans_id in visited:
                    continue
                visited.add(trans_id)

                trans = self._transformations.get(trans_id)
                if trans:
                    # Only include if this node is an output
                    if current_id in trans.output_node_ids:
                        transformations.append(trans)

                    # Traverse to inputs
                    for input_id in trans.input_node_ids:
                        _traverse(input_id)

        with self._lock:
            _traverse(node_id)

        # Reverse to get chronological order
        return list(reversed(transformations))

    def get_root_nodes(self) -> List[DataNode]:
        """Get all root nodes (no parents)."""
        with self._lock:
            return [n for n in self._nodes.values() if not n.parent_ids]

    def get_leaf_nodes(self) -> List[DataNode]:
        """Get all leaf nodes (no children)."""
        with self._lock:
            return [n for n in self._nodes.values() if not n.child_ids]

    def get_nodes_by_type(self, node_type: DataNodeType) -> List[DataNode]:
        """Get all nodes of a specific type."""
        with self._lock:
            return [n for n in self._nodes.values() if n.node_type == node_type.value]

    def get_nodes_by_tag(self, tag: str) -> List[DataNode]:
        """Get all nodes with a specific tag."""
        with self._lock:
            return [n for n in self._nodes.values() if tag in n.tags]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "transformation_count": len(self._transformations),
                "edge_count": len(self._edges),
                "root_count": len(self.get_root_nodes()),
                "leaf_count": len(self.get_leaf_nodes()),
                "nodes_by_type": {
                    t.value: len([n for n in self._nodes.values() if n.node_type == t.value])
                    for t in DataNodeType
                },
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        with self._lock:
            return {
                "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
                "transformations": {k: v.to_dict() for k, v in self._transformations.items()},
                "edges": {k: asdict(v) for k, v in self._edges.items()},
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageGraph":
        """Create graph from dictionary."""
        graph = cls()
        for node_data in data.get("nodes", {}).values():
            graph._nodes[node_data["node_id"]] = DataNode.from_dict(node_data)
        for trans_data in data.get("transformations", {}).values():
            graph._transformations[trans_data["transformation_id"]] = DataTransformation.from_dict(trans_data)
        for edge_data in data.get("edges", {}).values():
            graph._edges[edge_data["edge_id"]] = LineageEdge(**edge_data)
        return graph


# =============================================================================
# Main Lineage Tracker
# =============================================================================


class DataLineageTracker:
    """
    Data Lineage Tracking System for EU AI Act compliance.

    Tracks data provenance from source to model input/output, supporting:
    - Article 10 (Data Governance) - documenting data preparation processes
    - Article 11 (Technical Documentation) - Annex IV requirements
    - Article 12 (Record-Keeping) - data-related event logging

    Usage:
        config = DataLineageConfig()
        tracker = DataLineageTracker(config)

        # Register a data source
        source_node = tracker.register_source(
            name="binance_btcusdt",
            source_system="binance",
            source_location="api/v3/klines",
            df=raw_df,
        )

        # Track a transformation
        cleaned_node = tracker.track_transformation(
            transformation_type=TransformationType.OUTLIER_WINSORIZATION,
            input_nodes=[source_node],
            output_name="cleaned_btcusdt",
            output_df=cleaned_df,
            parameters={"percentiles": [1, 99]},
        )

        # Get lineage for a node
        lineage = tracker.get_lineage(cleaned_node.node_id)

        # Export for documentation
        tracker.export_lineage("lineage_report.json")
    """

    def __init__(self, config: Optional[DataLineageConfig] = None):
        """Initialize the tracker."""
        self.config = config or DataLineageConfig()

        # Graph
        self._graph = LineageGraph()

        # Storage
        self._storage_path = Path(self.config.storage_path)
        if self.config.enable_persistence:
            self._storage_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Active tracking context
        self._current_pipeline: Optional[str] = None

        logger.info("DataLineageTracker initialized")

    # =========================================================================
    # Source Registration
    # =========================================================================

    def register_source(
        self,
        name: str,
        source_system: str,
        source_location: str,
        df: Optional[pd.DataFrame] = None,
        extraction_method: str = "api_call",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> DataNode:
        """
        Register a data source.

        Per Article 10(2)(a), documents design choices and data collection processes.

        Args:
            name: Name of the data source
            source_system: System the data comes from (e.g., "binance", "alpaca")
            source_location: Location within the system (e.g., API endpoint, file path)
            df: Optional DataFrame to compute statistics from
            extraction_method: How data was extracted
            description: Description of the data source
            metadata: Additional metadata
            tags: Classification tags

        Returns:
            DataNode representing the source
        """
        node = DataNode(
            name=name,
            node_type=DataNodeType.RAW_SOURCE.value,
            description=description or f"Raw data from {source_system}",
            source_system=source_system,
            source_location=source_location,
            extraction_method=extraction_method,
            tags=tags or ["source", source_system],
            metadata=metadata or {},
        )

        # Populate from DataFrame if provided
        if df is not None:
            node.columns = list(df.columns)
            node.column_types = {col: str(df[col].dtype) for col in df.columns}
            node.record_count = len(df)

            if self.config.enable_content_hashing:
                node.content_hash = self._compute_content_hash(df)
                node.statistics_hash = self._compute_statistics_hash(df)

        with self._lock:
            self._graph.add_node(node)

        if self.config.enable_persistence:
            self._persist_node(node)

        logger.info(f"Registered source: {name} from {source_system}")
        return node

    # =========================================================================
    # Transformation Tracking
    # =========================================================================

    def track_transformation(
        self,
        transformation_type: TransformationType,
        input_nodes: List[DataNode],
        output_name: str,
        output_df: Optional[pd.DataFrame] = None,
        output_node_type: DataNodeType = DataNodeType.TRANSFORMED,
        parameters: Optional[Dict[str, Any]] = None,
        function_name: str = "",
        module_name: str = "",
        description: str = "",
        validate: bool = True,
        tags: Optional[List[str]] = None,
    ) -> DataNode:
        """
        Track a data transformation.

        Per Article 10(2)(b), documents preparation processes.

        Args:
            transformation_type: Type of transformation
            input_nodes: Input data nodes
            output_name: Name for the output node
            output_df: Optional output DataFrame
            output_node_type: Type of output node
            parameters: Transformation parameters
            function_name: Name of the function that performed the transformation
            module_name: Module containing the function
            description: Description of the transformation
            validate: Whether to validate the transformation
            tags: Classification tags

        Returns:
            DataNode representing the output
        """
        # Create output node
        output_node = DataNode(
            name=output_name,
            node_type=output_node_type.value,
            description=description or f"Output of {transformation_type.value}",
            parent_ids=[n.node_id for n in input_nodes],
            tags=tags or [transformation_type.value],
        )

        # Populate from DataFrame
        if output_df is not None:
            output_node.columns = list(output_df.columns)
            output_node.column_types = {col: str(output_df[col].dtype) for col in output_df.columns}
            output_node.record_count = len(output_df)

            if self.config.enable_content_hashing:
                output_node.content_hash = self._compute_content_hash(output_df)
                output_node.statistics_hash = self._compute_statistics_hash(output_df)

        # Create transformation record
        transformation = DataTransformation(
            transformation_type=transformation_type.value,
            name=f"{transformation_type.value}_{output_name}",
            description=description,
            input_node_ids=[n.node_id for n in input_nodes],
            output_node_ids=[output_node.node_id],
            parameters=parameters or {},
            function_name=function_name,
            module_name=module_name,
        )

        # Validate if requested
        if validate and self.config.enable_transformation_validation and output_df is not None:
            validation_results = self._validate_transformation(
                transformation, input_nodes, output_df
            )
            transformation.validated = validation_results.get("valid", False)
            transformation.validation_results = validation_results

        with self._lock:
            self._graph.add_node(output_node)
            self._graph.add_transformation(transformation)

        if self.config.enable_persistence:
            self._persist_node(output_node)
            self._persist_transformation(transformation)

        if self.config.log_all_transformations:
            logger.info(
                f"Tracked transformation: {transformation_type.value} -> {output_name} "
                f"(inputs: {[n.name for n in input_nodes]})"
            )

        return output_node

    def track_feature_shift(
        self,
        input_node: DataNode,
        output_name: str,
        output_df: pd.DataFrame,
        shift_periods: int = 1,
        shifted_columns: Optional[List[str]] = None,
    ) -> DataNode:
        """
        Track feature shifting transformation (critical for trading).

        This is essential for preventing look-ahead bias - all features
        must be shifted to ensure they represent information available
        BEFORE the current decision point.

        Args:
            input_node: Input data node
            output_name: Name for the output node
            output_df: Output DataFrame after shifting
            shift_periods: Number of periods shifted
            shifted_columns: Columns that were shifted

        Returns:
            DataNode representing the shifted data
        """
        return self.track_transformation(
            transformation_type=TransformationType.FEATURE_SHIFT,
            input_nodes=[input_node],
            output_name=output_name,
            output_df=output_df,
            parameters={
                "shift_periods": shift_periods,
                "shifted_columns": shifted_columns or [],
                "look_ahead_bias_prevention": True,
            },
            description=f"Feature shift by {shift_periods} periods to prevent look-ahead bias",
            tags=["feature_shift", "look_ahead_prevention"],
        )

    def track_train_test_split(
        self,
        input_node: DataNode,
        train_name: str,
        test_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_ratio: float = 0.8,
        split_method: str = "time_based",
        validation_name: Optional[str] = None,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[DataNode, DataNode, Optional[DataNode]]:
        """
        Track train/test split transformation.

        Args:
            input_node: Input data node
            train_name: Name for training set node
            test_name: Name for test set node
            train_df: Training DataFrame
            test_df: Test DataFrame
            split_ratio: Train/test split ratio
            split_method: Method used for splitting
            validation_name: Optional name for validation set
            val_df: Optional validation DataFrame

        Returns:
            Tuple of (train_node, test_node, optional_val_node)
        """
        # Track training set
        train_node = self.track_transformation(
            transformation_type=TransformationType.TRAIN_TEST_SPLIT,
            input_nodes=[input_node],
            output_name=train_name,
            output_df=train_df,
            output_node_type=DataNodeType.SPLIT,
            parameters={
                "split_ratio": split_ratio,
                "split_method": split_method,
                "split_type": "training",
            },
            tags=["training", "split"],
        )

        # Track test set
        test_node = self.track_transformation(
            transformation_type=TransformationType.TRAIN_TEST_SPLIT,
            input_nodes=[input_node],
            output_name=test_name,
            output_df=test_df,
            output_node_type=DataNodeType.SPLIT,
            parameters={
                "split_ratio": 1 - split_ratio,
                "split_method": split_method,
                "split_type": "testing",
            },
            tags=["testing", "split"],
        )

        # Track validation set if provided
        val_node = None
        if validation_name and val_df is not None:
            val_node = self.track_transformation(
                transformation_type=TransformationType.TRAIN_TEST_SPLIT,
                input_nodes=[input_node],
                output_name=validation_name,
                output_df=val_df,
                output_node_type=DataNodeType.SPLIT,
                parameters={
                    "split_method": split_method,
                    "split_type": "validation",
                },
                tags=["validation", "split"],
            )

        return train_node, test_node, val_node

    def track_normalization(
        self,
        input_node: DataNode,
        output_name: str,
        output_df: pd.DataFrame,
        method: str = "z_score",
        statistics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> DataNode:
        """
        Track normalization transformation.

        Args:
            input_node: Input data node
            output_name: Name for the output node
            output_df: Normalized DataFrame
            method: Normalization method (z_score, min_max, etc.)
            statistics: Statistics used for normalization

        Returns:
            DataNode representing the normalized data
        """
        return self.track_transformation(
            transformation_type=TransformationType.NORMALIZATION if method == "z_score" else TransformationType.STANDARDIZATION,
            input_nodes=[input_node],
            output_name=output_name,
            output_df=output_df,
            output_node_type=DataNodeType.NORMALIZED,
            parameters={
                "method": method,
                "statistics": statistics or {},
            },
            tags=["normalization", method],
        )

    def track_merge(
        self,
        input_nodes: List[DataNode],
        output_name: str,
        output_df: pd.DataFrame,
        merge_method: str = "concat",
        merge_on: Optional[List[str]] = None,
    ) -> DataNode:
        """
        Track data merge transformation.

        Args:
            input_nodes: List of input nodes to merge
            output_name: Name for the merged output
            output_df: Merged DataFrame
            merge_method: Method used (concat, merge, join)
            merge_on: Columns used for merging

        Returns:
            DataNode representing the merged data
        """
        return self.track_transformation(
            transformation_type=TransformationType.MERGE,
            input_nodes=input_nodes,
            output_name=output_name,
            output_df=output_df,
            output_node_type=DataNodeType.MERGED,
            parameters={
                "merge_method": merge_method,
                "merge_on": merge_on or [],
                "input_count": len(input_nodes),
            },
            tags=["merge", merge_method],
        )

    # =========================================================================
    # Lineage Queries
    # =========================================================================

    def get_node(self, node_id: str) -> Optional[DataNode]:
        """Get a node by ID."""
        return self._graph.get_node(node_id)

    def get_node_by_name(self, name: str) -> Optional[DataNode]:
        """Get a node by name."""
        with self._lock:
            for node in self._graph._nodes.values():
                if node.name == name:
                    return node
        return None

    def get_lineage(self, node_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for a node.

        Returns upstream lineage (ancestors) and the transformation chain.
        """
        node = self._graph.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        ancestors = self._graph.get_ancestors(node_id, self.config.max_lineage_depth)
        transformations = self._graph.get_transformation_chain(node_id)

        return {
            "node": node.to_dict(),
            "ancestors": [a.to_dict() for a in ancestors],
            "transformation_chain": [t.to_dict() for t in transformations],
            "lineage_depth": len(ancestors),
        }

    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """
        Get impact analysis for a node.

        Returns downstream lineage (what depends on this node).
        """
        node = self._graph.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        descendants = self._graph.get_descendants(node_id, self.config.max_lineage_depth)

        return {
            "node": node.to_dict(),
            "affected_nodes": [d.to_dict() for d in descendants],
            "affected_count": len(descendants),
        }

    def get_transformation_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get the history of transformations that produced a node."""
        transformations = self._graph.get_transformation_chain(node_id)
        return [t.to_dict() for t in transformations]

    # =========================================================================
    # Export and Reporting
    # =========================================================================

    def export_lineage(self, output_path: str, format: str = "json") -> None:
        """
        Export lineage graph for documentation.

        Supports Article 11/Annex IV documentation requirements.

        Args:
            output_path: Path to save the export
            format: Export format (json, graphml)
        """
        if format == "json":
            export_data = {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "statistics": self._graph.get_statistics(),
                "graph": self._graph.to_dict(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Lineage exported to {output_path}")

    def generate_documentation(self) -> Dict[str, Any]:
        """
        Generate documentation for technical documentation (Annex IV).

        Returns structured documentation of:
        - Data sources and collection methods
        - Preparation processes (transformations)
        - Data quality and validation
        """
        root_nodes = self._graph.get_root_nodes()
        leaf_nodes = self._graph.get_leaf_nodes()

        doc = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_sources": {
                "count": len(root_nodes),
                "sources": [
                    {
                        "name": n.name,
                        "system": n.source_system,
                        "location": n.source_location,
                        "extraction_method": n.extraction_method,
                        "record_count": n.record_count,
                    }
                    for n in root_nodes
                ],
            },
            "preparation_processes": {
                "transformation_count": len(self._graph._transformations),
                "transformations_by_type": {},
            },
            "model_inputs": {
                "count": len([n for n in leaf_nodes if n.node_type == DataNodeType.MODEL_INPUT.value]),
                "inputs": [
                    {
                        "name": n.name,
                        "columns": n.columns,
                        "record_count": n.record_count,
                        "lineage_depth": len(self._graph.get_ancestors(n.node_id)),
                    }
                    for n in leaf_nodes
                    if n.node_type == DataNodeType.MODEL_INPUT.value
                ],
            },
            "statistics": self._graph.get_statistics(),
        }

        # Count transformations by type
        for trans in self._graph._transformations.values():
            t_type = trans.transformation_type
            doc["preparation_processes"]["transformations_by_type"][t_type] = \
                doc["preparation_processes"]["transformations_by_type"].get(t_type, 0) + 1

        return doc

    def generate_lineage_report(self, node_id: str) -> str:
        """
        Generate a human-readable lineage report for a node.

        Useful for technical documentation.
        """
        lineage = self.get_lineage(node_id)
        if "error" in lineage:
            return f"Error: {lineage['error']}"

        report_lines = [
            "=" * 60,
            f"DATA LINEAGE REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            f"Target Node: {lineage['node']['name']} ({lineage['node']['node_id']})",
            f"Type: {lineage['node']['node_type']}",
            f"Records: {lineage['node']['record_count']}",
            f"Columns: {len(lineage['node']['columns'])}",
            "",
            "-" * 40,
            "TRANSFORMATION CHAIN",
            "-" * 40,
        ]

        for i, trans in enumerate(lineage["transformation_chain"], 1):
            report_lines.append(f"{i}. {trans['transformation_type']}")
            report_lines.append(f"   Name: {trans['name']}")
            report_lines.append(f"   Executed: {trans['executed_at']}")
            if trans['parameters']:
                report_lines.append(f"   Parameters: {json.dumps(trans['parameters'], default=str)[:100]}...")
            report_lines.append("")

        report_lines.extend([
            "-" * 40,
            "ANCESTOR NODES",
            "-" * 40,
        ])

        for ancestor in lineage["ancestors"]:
            report_lines.append(f"- {ancestor['name']} ({ancestor['node_type']})")

        report_lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(report_lines)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[str] = None) -> None:
        """Save the lineage graph to disk."""
        save_path = Path(path) if path else self._storage_path / "lineage_graph.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self._graph.to_dict(), f, indent=2, default=str)

        logger.info(f"Lineage graph saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load the lineage graph from disk."""
        load_path = Path(path) if path else self._storage_path / "lineage_graph.json"

        if not load_path.exists():
            logger.warning(f"Lineage file not found: {load_path}")
            return

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._graph = LineageGraph.from_dict(data)

        logger.info(f"Lineage graph loaded from {load_path}")

    def _persist_node(self, node: DataNode) -> None:
        """Persist a node to storage."""
        if not self.config.enable_persistence:
            return

        nodes_path = self._storage_path / "nodes"
        nodes_path.mkdir(exist_ok=True)

        with open(nodes_path / f"{node.node_id}.json", "w", encoding="utf-8") as f:
            json.dump(node.to_dict(), f, indent=2, default=str)

    def _persist_transformation(self, transformation: DataTransformation) -> None:
        """Persist a transformation to storage."""
        if not self.config.enable_persistence:
            return

        trans_path = self._storage_path / "transformations"
        trans_path.mkdir(exist_ok=True)

        with open(trans_path / f"{transformation.transformation_id}.json", "w", encoding="utf-8") as f:
            json.dump(transformation.to_dict(), f, indent=2, default=str)

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_transformation(
        self,
        transformation: DataTransformation,
        input_nodes: List[DataNode],
        output_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Validate a transformation."""
        results = {
            "valid": True,
            "checks": [],
        }

        # Row count validation
        if self.config.validate_row_count_changes:
            total_input_rows = sum(n.record_count for n in input_nodes)
            output_rows = len(output_df)

            # Some transformations should maintain row count
            maintain_count_types = [
                TransformationType.NORMALIZATION.value,
                TransformationType.STANDARDIZATION.value,
                TransformationType.FEATURE_ENGINEERING.value,
            ]

            if transformation.transformation_type in maintain_count_types:
                if output_rows != total_input_rows:
                    results["checks"].append({
                        "check": "row_count_maintained",
                        "passed": False,
                        "expected": total_input_rows,
                        "actual": output_rows,
                    })
                    results["valid"] = False

        # Column validation
        if self.config.validate_column_changes:
            # Check for unexpected column loss
            if input_nodes and input_nodes[0].columns:
                expected_cols = set(input_nodes[0].columns)
                actual_cols = set(output_df.columns)

                # For merges, columns should increase or stay same
                if transformation.transformation_type == TransformationType.MERGE.value:
                    all_input_cols = set()
                    for node in input_nodes:
                        all_input_cols.update(node.columns)

                    if not all_input_cols.issubset(actual_cols):
                        missing = all_input_cols - actual_cols
                        results["checks"].append({
                            "check": "merge_columns_preserved",
                            "passed": False,
                            "missing_columns": list(missing),
                        })
                        results["valid"] = False

        return results

    # =========================================================================
    # Hashing
    # =========================================================================

    def _compute_content_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of DataFrame content."""
        if len(df) == 0:
            return hashlib.sha256(b"empty").hexdigest()[:16]

        # Sample if too large
        if len(df) > self.config.hash_sample_size:
            sample = df.sample(self.config.hash_sample_size, random_state=42)
        else:
            sample = df

        # Reset index to handle DataFrames with non-unique indices (e.g., after merge)
        sample = sample.reset_index(drop=True)

        # Create hash from values
        content = sample.to_json(date_format="iso", default_handler=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_statistics_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of DataFrame statistics."""
        stats = {
            "shape": df.shape,
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            stats[f"{col}_mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            stats[f"{col}_std"] = float(df[col].std()) if not df[col].isnull().all() else None

        content = json.dumps(stats, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Factory Functions
# =============================================================================


def create_data_lineage_tracker(
    config: Optional[Union[Dict[str, Any], DataLineageConfig]] = None,
) -> DataLineageTracker:
    """
    Create a DataLineageTracker instance.

    Args:
        config: Optional configuration dictionary or DataLineageConfig

    Returns:
        Configured DataLineageTracker instance
    """
    if config is None:
        return DataLineageTracker()
    elif isinstance(config, DataLineageConfig):
        return DataLineageTracker(config)
    else:
        lineage_config = DataLineageConfig(
            storage_path=config.get("storage_path", "logs/ai_act/data_lineage"),
            enable_persistence=config.get("enable_persistence", True),
            enable_content_hashing=config.get("enable_content_hashing", True),
            hash_sample_size=config.get("hash_sample_size", 10000),
            enable_transformation_validation=config.get("enable_transformation_validation", True),
            validate_row_count_changes=config.get("validate_row_count_changes", True),
            validate_column_changes=config.get("validate_column_changes", True),
            max_lineage_depth=config.get("max_lineage_depth", 100),
            log_all_transformations=config.get("log_all_transformations", True),
        )
        return DataLineageTracker(lineage_config)
