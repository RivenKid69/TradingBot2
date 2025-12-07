# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Data Lineage Tracking System.

Tests cover:
- DataLineageTracker - Main lineage tracking
- DataNode - Data artifact representation
- DataTransformation - Transformation records
- LineageGraph - Graph representation
- Factory functions
- Thread safety
"""

import json
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from services.ai_act.data_lineage import (
    # Enums
    DataNodeType,
    TransformationType,
    # Data structures
    DataNode,
    DataTransformation,
    LineageEdge,
    # Configuration
    DataLineageConfig,
    # Main classes
    LineageGraph,
    DataLineageTracker,
    # Factory functions
    create_data_lineage_tracker,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def lineage_config(temp_storage_dir):
    """Create a test lineage configuration."""
    return DataLineageConfig(
        storage_path=temp_storage_dir,
        enable_persistence=True,
        enable_content_hashing=True,
        hash_sample_size=100,
        enable_transformation_validation=True,
        validate_row_count_changes=True,
        validate_column_changes=True,
        max_lineage_depth=50,
        log_all_transformations=True,
    )


@pytest.fixture
def sample_df():
    """Create a sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "symbol": np.random.choice(["BTCUSDT", "ETHUSDT"], 100),
        "open": 50000 + np.random.randn(100) * 100,
        "high": 50050 + np.random.randn(100) * 100,
        "low": 49950 + np.random.randn(100) * 100,
        "close": 50000 + np.random.randn(100) * 100,
        "volume": np.abs(np.random.randn(100) * 1000 + 5000),
    })


@pytest.fixture
def normalized_df(sample_df):
    """Create a normalized version of sample DataFrame."""
    df = sample_df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


@pytest.fixture
def lineage_tracker(lineage_config):
    """Create a test lineage tracker."""
    return DataLineageTracker(lineage_config)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDataNodeType:
    """Tests for DataNodeType enum."""

    def test_node_type_values(self):
        """Test all node type values."""
        assert DataNodeType.RAW_SOURCE.value == "raw_source"
        assert DataNodeType.EXTRACTED.value == "extracted"
        assert DataNodeType.CLEANED.value == "cleaned"
        assert DataNodeType.TRANSFORMED.value == "transformed"
        assert DataNodeType.NORMALIZED.value == "normalized"
        assert DataNodeType.MERGED.value == "merged"
        assert DataNodeType.SPLIT.value == "split"
        assert DataNodeType.MODEL_INPUT.value == "model_input"


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_cleaning_types(self):
        """Test cleaning transformation types."""
        assert TransformationType.MISSING_VALUE_IMPUTATION.value == "missing_value_imputation"
        assert TransformationType.OUTLIER_WINSORIZATION.value == "outlier_winsorization"
        assert TransformationType.DUPLICATE_REMOVAL.value == "duplicate_removal"

    def test_transformation_types(self):
        """Test transformation types."""
        assert TransformationType.NORMALIZATION.value == "normalization"
        assert TransformationType.FEATURE_ENGINEERING.value == "feature_engineering"
        assert TransformationType.ROLLING_WINDOW.value == "rolling_window"

    def test_critical_types(self):
        """Test critical transformation types for trading."""
        assert TransformationType.FEATURE_SHIFT.value == "feature_shift"
        assert TransformationType.TRAIN_TEST_SPLIT.value == "train_test_split"
        assert TransformationType.WALK_FORWARD_SPLIT.value == "walk_forward_split"


# =============================================================================
# DataNode Tests
# =============================================================================


class TestDataNode:
    """Tests for DataNode dataclass."""

    def test_create_node(self):
        """Test creating a data node."""
        node = DataNode(
            name="raw_data",
            node_type=DataNodeType.RAW_SOURCE.value,
            source_system="binance",
        )

        assert node.node_id.startswith("DN-")
        assert node.name == "raw_data"
        assert node.source_system == "binance"

    def test_node_auto_id(self):
        """Test node auto-generates ID."""
        node = DataNode()

        assert node.node_id.startswith("DN-")
        assert len(node.node_id) == 15  # DN- + 12 hex chars

    def test_node_auto_timestamp(self):
        """Test node auto-generates timestamps."""
        node = DataNode()

        assert node.created_at != ""
        assert node.updated_at != ""

    def test_node_to_dict(self):
        """Test node serialization."""
        node = DataNode(
            name="test",
            columns=["col1", "col2"],
            record_count=100,
        )

        data = node.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["columns"] == ["col1", "col2"]

    def test_node_from_dict(self):
        """Test node deserialization."""
        data = {
            "node_id": "DN-TEST123",
            "name": "test_node",
            "node_type": "raw_source",
            "columns": ["a", "b", "c"],
        }

        node = DataNode.from_dict(data)

        assert node.node_id == "DN-TEST123"
        assert node.name == "test_node"
        assert len(node.columns) == 3

    def test_node_with_relationships(self):
        """Test node with parent/child relationships."""
        node = DataNode(
            name="child",
            parent_ids=["DN-PARENT1", "DN-PARENT2"],
            child_ids=["DN-CHILD1"],
        )

        assert len(node.parent_ids) == 2
        assert len(node.child_ids) == 1


# =============================================================================
# DataTransformation Tests
# =============================================================================


class TestDataTransformation:
    """Tests for DataTransformation dataclass."""

    def test_create_transformation(self):
        """Test creating a transformation."""
        trans = DataTransformation(
            transformation_type=TransformationType.NORMALIZATION.value,
            name="normalize_prices",
            input_node_ids=["DN-INPUT"],
            output_node_ids=["DN-OUTPUT"],
        )

        assert trans.transformation_id.startswith("DT-")
        assert trans.transformation_type == "normalization"

    def test_transformation_auto_id(self):
        """Test transformation auto-generates ID."""
        trans = DataTransformation()

        assert trans.transformation_id.startswith("DT-")

    def test_transformation_with_parameters(self):
        """Test transformation with parameters."""
        trans = DataTransformation(
            transformation_type="normalization",
            parameters={
                "method": "z_score",
                "columns": ["open", "high", "low", "close"],
            },
        )

        assert trans.parameters["method"] == "z_score"

    def test_transformation_to_dict(self):
        """Test transformation serialization."""
        trans = DataTransformation(
            transformation_type="test",
            parameters={"key": "value"},
        )

        data = trans.to_dict()

        assert isinstance(data, dict)
        assert data["parameters"]["key"] == "value"

    def test_transformation_from_dict(self):
        """Test transformation deserialization."""
        data = {
            "transformation_id": "DT-TEST123",
            "transformation_type": "normalization",
            "name": "test_transform",
        }

        trans = DataTransformation.from_dict(data)

        assert trans.transformation_id == "DT-TEST123"

    def test_transformation_validation(self):
        """Test transformation validation fields."""
        trans = DataTransformation(
            validated=True,
            validation_results={"row_count_check": True},
        )

        assert trans.validated is True
        assert "row_count_check" in trans.validation_results


# =============================================================================
# LineageEdge Tests
# =============================================================================


class TestLineageEdge:
    """Tests for LineageEdge dataclass."""

    def test_create_edge(self):
        """Test creating a lineage edge."""
        edge = LineageEdge(
            source_node_id="DN-SOURCE",
            target_node_id="DN-TARGET",
            transformation_id="DT-TRANS",
        )

        assert edge.edge_id.startswith("LE-")
        assert edge.source_node_id == "DN-SOURCE"
        assert edge.target_node_id == "DN-TARGET"


# =============================================================================
# LineageGraph Tests
# =============================================================================


class TestLineageGraph:
    """Tests for LineageGraph class."""

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = LineageGraph()
        node = DataNode(name="test")

        graph.add_node(node)

        assert graph.get_node(node.node_id) is not None

    def test_add_transformation(self):
        """Test adding a transformation to the graph."""
        graph = LineageGraph()

        # Add input and output nodes
        input_node = DataNode(name="input")
        output_node = DataNode(name="output")
        graph.add_node(input_node)
        graph.add_node(output_node)

        # Add transformation
        trans = DataTransformation(
            input_node_ids=[input_node.node_id],
            output_node_ids=[output_node.node_id],
        )
        graph.add_transformation(trans)

        # Check transformation was added
        assert graph.get_transformation(trans.transformation_id) is not None

        # Check relationships were updated
        input_retrieved = graph.get_node(input_node.node_id)
        assert output_node.node_id in input_retrieved.child_ids

        output_retrieved = graph.get_node(output_node.node_id)
        assert input_node.node_id in output_retrieved.parent_ids

    def test_get_ancestors(self):
        """Test getting ancestor nodes."""
        graph = LineageGraph()

        # Create a chain: grandparent -> parent -> child
        grandparent = DataNode(name="grandparent")
        parent = DataNode(name="parent", parent_ids=[grandparent.node_id])
        child = DataNode(name="child", parent_ids=[parent.node_id])

        grandparent.child_ids = [parent.node_id]
        parent.child_ids = [child.node_id]

        graph.add_node(grandparent)
        graph.add_node(parent)
        graph.add_node(child)

        ancestors = graph.get_ancestors(child.node_id)

        assert len(ancestors) == 2
        ancestor_names = [a.name for a in ancestors]
        assert "parent" in ancestor_names
        assert "grandparent" in ancestor_names

    def test_get_descendants(self):
        """Test getting descendant nodes."""
        graph = LineageGraph()

        # Create a chain: parent -> child -> grandchild
        parent = DataNode(name="parent")
        child = DataNode(name="child", parent_ids=[parent.node_id])
        grandchild = DataNode(name="grandchild", parent_ids=[child.node_id])

        parent.child_ids = [child.node_id]
        child.child_ids = [grandchild.node_id]

        graph.add_node(parent)
        graph.add_node(child)
        graph.add_node(grandchild)

        descendants = graph.get_descendants(parent.node_id)

        assert len(descendants) == 2

    def test_get_root_nodes(self):
        """Test getting root nodes (no parents)."""
        graph = LineageGraph()

        root1 = DataNode(name="root1")
        root2 = DataNode(name="root2")
        child = DataNode(name="child", parent_ids=[root1.node_id, root2.node_id])

        graph.add_node(root1)
        graph.add_node(root2)
        graph.add_node(child)

        roots = graph.get_root_nodes()

        assert len(roots) == 2

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes (no children)."""
        graph = LineageGraph()

        parent = DataNode(name="parent")
        leaf1 = DataNode(name="leaf1", parent_ids=[parent.node_id])
        leaf2 = DataNode(name="leaf2", parent_ids=[parent.node_id])

        parent.child_ids = [leaf1.node_id, leaf2.node_id]

        graph.add_node(parent)
        graph.add_node(leaf1)
        graph.add_node(leaf2)

        leaves = graph.get_leaf_nodes()

        assert len(leaves) == 2

    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        graph = LineageGraph()

        source = DataNode(name="source", node_type=DataNodeType.RAW_SOURCE.value)
        transformed = DataNode(name="transformed", node_type=DataNodeType.TRANSFORMED.value)

        graph.add_node(source)
        graph.add_node(transformed)

        sources = graph.get_nodes_by_type(DataNodeType.RAW_SOURCE)

        assert len(sources) == 1
        assert sources[0].name == "source"

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = LineageGraph()

        graph.add_node(DataNode(name="node1"))
        graph.add_node(DataNode(name="node2"))

        stats = graph.get_statistics()

        assert stats["node_count"] == 2
        assert "transformation_count" in stats

    def test_to_dict_from_dict(self):
        """Test graph serialization and deserialization."""
        graph = LineageGraph()

        node1 = DataNode(name="node1")
        node2 = DataNode(name="node2", parent_ids=[node1.node_id])
        node1.child_ids = [node2.node_id]

        graph.add_node(node1)
        graph.add_node(node2)

        # Serialize
        data = graph.to_dict()

        # Deserialize
        new_graph = LineageGraph.from_dict(data)

        assert new_graph.get_node(node1.node_id) is not None
        assert new_graph.get_node(node2.node_id) is not None


# =============================================================================
# DataLineageConfig Tests
# =============================================================================


class TestDataLineageConfig:
    """Tests for DataLineageConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataLineageConfig()

        assert config.storage_path == "logs/ai_act/data_lineage"
        assert config.enable_persistence is True
        assert config.enable_content_hashing is True
        assert config.max_lineage_depth == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataLineageConfig(
            storage_path="/custom/path",
            max_lineage_depth=50,
        )

        assert config.storage_path == "/custom/path"
        assert config.max_lineage_depth == 50


# =============================================================================
# DataLineageTracker Tests
# =============================================================================


class TestDataLineageTracker:
    """Tests for DataLineageTracker class."""

    def test_tracker_initialization(self, lineage_tracker):
        """Test tracker initialization."""
        assert lineage_tracker is not None

    def test_register_source(self, lineage_tracker, sample_df):
        """Test registering a data source."""
        node = lineage_tracker.register_source(
            name="binance_btcusdt",
            source_system="binance",
            source_location="api/v3/klines",
            df=sample_df,
            extraction_method="api_call",
        )

        assert node is not None
        assert node.name == "binance_btcusdt"
        assert node.node_type == DataNodeType.RAW_SOURCE.value
        assert node.source_system == "binance"
        assert node.record_count == len(sample_df)

    def test_register_source_with_metadata(self, lineage_tracker, sample_df):
        """Test registering a source with metadata."""
        node = lineage_tracker.register_source(
            name="test_source",
            source_system="test",
            source_location="test/path",
            df=sample_df,
            metadata={"api_version": "v3"},
            tags=["crypto", "binance"],
        )

        assert node.metadata["api_version"] == "v3"
        assert "crypto" in node.tags

    def test_track_transformation(self, lineage_tracker, sample_df, normalized_df):
        """Test tracking a transformation."""
        source_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        output_node = lineage_tracker.track_transformation(
            transformation_type=TransformationType.NORMALIZATION,
            input_nodes=[source_node],
            output_name="normalized_data",
            output_df=normalized_df,
            parameters={"method": "z_score"},
        )

        assert output_node is not None
        assert output_node.name == "normalized_data"
        assert source_node.node_id in output_node.parent_ids

    def test_track_feature_shift(self, lineage_tracker, sample_df):
        """Test tracking feature shift transformation."""
        source_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Simulate shift
        shifted_df = sample_df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            shifted_df[col] = shifted_df[col].shift(1)
        shifted_df = shifted_df.dropna()

        output_node = lineage_tracker.track_feature_shift(
            input_node=source_node,
            output_name="shifted_data",
            output_df=shifted_df,
            shift_periods=1,
            shifted_columns=["open", "high", "low", "close", "volume"],
        )

        assert output_node is not None
        assert "look_ahead_prevention" in output_node.tags

    def test_track_train_test_split(self, lineage_tracker, sample_df):
        """Test tracking train/test split."""
        source_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Split
        split_idx = int(len(sample_df) * 0.8)
        train_df = sample_df.iloc[:split_idx]
        test_df = sample_df.iloc[split_idx:]

        train_node, test_node, _ = lineage_tracker.track_train_test_split(
            input_node=source_node,
            train_name="training_data",
            test_name="test_data",
            train_df=train_df,
            test_df=test_df,
            split_ratio=0.8,
        )

        assert train_node is not None
        assert test_node is not None
        assert train_node.record_count == len(train_df)
        assert test_node.record_count == len(test_df)

    def test_track_train_val_test_split(self, lineage_tracker, sample_df):
        """Test tracking train/val/test split."""
        source_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Split
        train_df = sample_df.iloc[:60]
        val_df = sample_df.iloc[60:80]
        test_df = sample_df.iloc[80:]

        train_node, test_node, val_node = lineage_tracker.track_train_test_split(
            input_node=source_node,
            train_name="train",
            test_name="test",
            train_df=train_df,
            test_df=test_df,
            validation_name="val",
            val_df=val_df,
        )

        assert val_node is not None
        assert val_node.record_count == len(val_df)

    def test_track_normalization(self, lineage_tracker, sample_df):
        """Test tracking normalization."""
        source_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Normalize
        normalized_df = sample_df.copy()
        for col in ["open", "high", "low", "close"]:
            mean = normalized_df[col].mean()
            std = normalized_df[col].std()
            normalized_df[f"{col}_z"] = (normalized_df[col] - mean) / std

        output_node = lineage_tracker.track_normalization(
            input_node=source_node,
            output_name="normalized",
            output_df=normalized_df,
            method="z_score",
        )

        assert output_node is not None
        assert output_node.node_type == DataNodeType.NORMALIZED.value

    def test_track_merge(self, lineage_tracker, sample_df):
        """Test tracking merge operation."""
        df1 = sample_df.iloc[:50].copy()
        df2 = sample_df.iloc[50:].copy()

        node1 = lineage_tracker.register_source(
            name="source1",
            source_system="test",
            source_location="test1",
            df=df1,
        )

        node2 = lineage_tracker.register_source(
            name="source2",
            source_system="test",
            source_location="test2",
            df=df2,
        )

        merged_df = pd.concat([df1, df2])

        merged_node = lineage_tracker.track_merge(
            input_nodes=[node1, node2],
            output_name="merged_data",
            output_df=merged_df,
            merge_method="concat",
        )

        assert merged_node is not None
        assert merged_node.node_type == DataNodeType.MERGED.value
        assert len(merged_node.parent_ids) == 2

    def test_get_node(self, lineage_tracker, sample_df):
        """Test getting a node by ID."""
        node = lineage_tracker.register_source(
            name="test",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        retrieved = lineage_tracker.get_node(node.node_id)

        assert retrieved is not None
        assert retrieved.node_id == node.node_id

    def test_get_node_by_name(self, lineage_tracker, sample_df):
        """Test getting a node by name."""
        lineage_tracker.register_source(
            name="unique_name",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        retrieved = lineage_tracker.get_node_by_name("unique_name")

        assert retrieved is not None
        assert retrieved.name == "unique_name"

    def test_get_lineage(self, lineage_tracker, sample_df):
        """Test getting lineage for a node."""
        # Create a lineage chain
        source = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        cleaned = lineage_tracker.track_transformation(
            transformation_type=TransformationType.DUPLICATE_REMOVAL,
            input_nodes=[source],
            output_name="cleaned",
            output_df=sample_df,
        )

        normalized = lineage_tracker.track_transformation(
            transformation_type=TransformationType.NORMALIZATION,
            input_nodes=[cleaned],
            output_name="normalized",
            output_df=sample_df,
        )

        lineage = lineage_tracker.get_lineage(normalized.node_id)

        assert "node" in lineage
        assert "ancestors" in lineage
        assert "transformation_chain" in lineage
        assert len(lineage["ancestors"]) >= 2

    def test_get_impact_analysis(self, lineage_tracker, sample_df):
        """Test getting impact analysis."""
        source = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        child1 = lineage_tracker.track_transformation(
            transformation_type=TransformationType.NORMALIZATION,
            input_nodes=[source],
            output_name="child1",
            output_df=sample_df,
        )

        child2 = lineage_tracker.track_transformation(
            transformation_type=TransformationType.FEATURE_ENGINEERING,
            input_nodes=[source],
            output_name="child2",
            output_df=sample_df,
        )

        impact = lineage_tracker.get_impact_analysis(source.node_id)

        assert "affected_nodes" in impact
        assert impact["affected_count"] >= 2

    def test_get_transformation_history(self, lineage_tracker, sample_df):
        """Test getting transformation history."""
        source = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        cleaned = lineage_tracker.track_transformation(
            transformation_type=TransformationType.OUTLIER_WINSORIZATION,
            input_nodes=[source],
            output_name="cleaned",
            output_df=sample_df,
        )

        history = lineage_tracker.get_transformation_history(cleaned.node_id)

        assert len(history) >= 1

    def test_generate_documentation(self, lineage_tracker, sample_df):
        """Test generating documentation."""
        source = lineage_tracker.register_source(
            name="binance_data",
            source_system="binance",
            source_location="api/v3/klines",
            df=sample_df,
        )

        lineage_tracker.track_transformation(
            transformation_type=TransformationType.NORMALIZATION,
            input_nodes=[source],
            output_name="normalized",
            output_df=sample_df,
            output_node_type=DataNodeType.MODEL_INPUT,
        )

        doc = lineage_tracker.generate_documentation()

        assert "data_sources" in doc
        assert "preparation_processes" in doc
        assert doc["data_sources"]["count"] >= 1

    def test_generate_lineage_report(self, lineage_tracker, sample_df):
        """Test generating lineage report."""
        source = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        cleaned = lineage_tracker.track_transformation(
            transformation_type=TransformationType.OUTLIER_WINSORIZATION,
            input_nodes=[source],
            output_name="cleaned",
            output_df=sample_df,
        )

        report = lineage_tracker.generate_lineage_report(cleaned.node_id)

        assert "DATA LINEAGE REPORT" in report
        assert "TRANSFORMATION CHAIN" in report

    def test_export_lineage(self, lineage_tracker, sample_df, temp_storage_dir):
        """Test exporting lineage."""
        lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        output_path = f"{temp_storage_dir}/lineage_export.json"
        lineage_tracker.export_lineage(output_path)

        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "graph" in data
        assert "statistics" in data

    def test_save_and_load(self, lineage_config, sample_df, temp_storage_dir):
        """Test saving and loading lineage."""
        tracker1 = DataLineageTracker(lineage_config)

        node = tracker1.register_source(
            name="test_source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        save_path = f"{temp_storage_dir}/lineage_save.json"
        tracker1.save(save_path)

        # Create new tracker and load
        tracker2 = DataLineageTracker(lineage_config)
        tracker2.load(save_path)

        # Should be able to retrieve the node
        retrieved = tracker2.get_node(node.node_id)
        assert retrieved is not None
        assert retrieved.name == "test_source"


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_data_lineage_tracker_default(self):
        """Test creating tracker with defaults."""
        tracker = create_data_lineage_tracker()

        assert tracker is not None

    def test_create_data_lineage_tracker_with_config(self, lineage_config):
        """Test creating tracker with config object."""
        tracker = create_data_lineage_tracker(lineage_config)

        assert tracker is not None

    def test_create_data_lineage_tracker_with_dict(self, temp_storage_dir):
        """Test creating tracker with dict config."""
        config = {
            "storage_path": temp_storage_dir,
            "max_lineage_depth": 25,
        }
        tracker = create_data_lineage_tracker(config)

        assert tracker is not None


# =============================================================================
# Validation Tests
# =============================================================================


class TestTransformationValidation:
    """Tests for transformation validation."""

    def test_validate_row_count_maintained(self, lineage_tracker, sample_df):
        """Test validation of row count maintenance."""
        source = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Normalization should maintain row count
        output = lineage_tracker.track_transformation(
            transformation_type=TransformationType.NORMALIZATION,
            input_nodes=[source],
            output_name="normalized",
            output_df=sample_df,  # Same row count
            validate=True,
        )

        # Should pass validation
        assert output is not None

    def test_validate_column_preservation(self, lineage_tracker, sample_df):
        """Test validation of column preservation in merges."""
        df1 = sample_df.copy()
        df2 = sample_df.copy()
        df2["extra_col"] = 1

        node1 = lineage_tracker.register_source(
            name="source1",
            source_system="test",
            source_location="test1",
            df=df1,
        )

        node2 = lineage_tracker.register_source(
            name="source2",
            source_system="test",
            source_location="test2",
            df=df2,
        )

        # Merge that preserves columns
        merged_df = pd.concat([df1, df2])

        merged = lineage_tracker.track_merge(
            input_nodes=[node1, node2],
            output_name="merged",
            output_df=merged_df,
        )

        assert merged is not None


# =============================================================================
# Hashing Tests
# =============================================================================


class TestHashing:
    """Tests for content and statistics hashing."""

    def test_content_hash_computed(self, lineage_tracker, sample_df):
        """Test that content hash is computed."""
        node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        assert node.content_hash != ""
        assert len(node.content_hash) == 16

    def test_statistics_hash_computed(self, lineage_tracker, sample_df):
        """Test that statistics hash is computed."""
        node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        assert node.statistics_hash != ""

    def test_different_data_different_hash(self, lineage_tracker, sample_df):
        """Test that different data produces different hashes."""
        node1 = lineage_tracker.register_source(
            name="source1",
            source_system="test",
            source_location="test1",
            df=sample_df,
        )

        # Modify data
        different_df = sample_df.copy()
        different_df["close"] = different_df["close"] * 2

        node2 = lineage_tracker.register_source(
            name="source2",
            source_system="test",
            source_location="test2",
            df=different_df,
        )

        assert node1.content_hash != node2.content_hash


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registrations(self, lineage_tracker, sample_df):
        """Test concurrent source registrations."""
        nodes = []
        errors = []

        def register_source(index):
            try:
                node = lineage_tracker.register_source(
                    name=f"source_{index}",
                    source_system="test",
                    source_location=f"test_{index}",
                    df=sample_df,
                )
                nodes.append(node)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=register_source, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(nodes) == 10


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, lineage_tracker):
        """Test handling empty DataFrame."""
        df = pd.DataFrame()

        node = lineage_tracker.register_source(
            name="empty",
            source_system="test",
            source_location="test",
            df=df,
        )

        assert node is not None
        assert node.record_count == 0

    def test_get_nonexistent_node(self, lineage_tracker):
        """Test getting nonexistent node."""
        result = lineage_tracker.get_node("DN-NONEXISTENT")

        assert result is None

    def test_lineage_for_nonexistent_node(self, lineage_tracker):
        """Test getting lineage for nonexistent node."""
        lineage = lineage_tracker.get_lineage("DN-NONEXISTENT")

        assert "error" in lineage

    def test_deep_lineage(self, lineage_tracker, sample_df):
        """Test handling deep lineage chains."""
        current_node = lineage_tracker.register_source(
            name="source",
            source_system="test",
            source_location="test",
            df=sample_df,
        )

        # Create a deep chain
        for i in range(20):
            current_node = lineage_tracker.track_transformation(
                transformation_type=TransformationType.CUSTOM,
                input_nodes=[current_node],
                output_name=f"step_{i}",
                output_df=sample_df,
            )

        lineage = lineage_tracker.get_lineage(current_node.node_id)

        assert lineage["lineage_depth"] == 20
