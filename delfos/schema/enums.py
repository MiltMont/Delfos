"""Enumerations for the Cue-Tag-Content graph schema.

These define the closed vocabularies used by the node and edge models. They are
the code-specific instantiation of the abstract Cue-Tag-Content design from the
paper "Memory is Reconstructed, Not Retrieved" (arXiv 2606.06036).
"""

from enum import StrEnum


class NodeType(StrEnum):
    """Discriminator for the three node kinds in the graph."""

    CUE = "cue"
    TAG = "tag"
    CONTENT = "content"


class NodeStatus(StrEnum):
    """Lifecycle status of a node.

    ``DELETED`` marks a tombstone: the node is retained (with its cues and
    edges) so agents asking about historical behaviour still find it, but it no
    longer reflects code present at ``HEAD``.
    """

    ACTIVE = "active"
    DELETED = "deleted"


class MemoryLayer(StrEnum):
    """The three memory layers, mapped onto code.

    - ``EPISODIC``: concrete change events (commits, PRs).
    - ``SEMANTIC``: the stable exported API surface (interfaces, types, constants).
    - ``TOPIC``: modules and packages as architectural clusters.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    TOPIC = "topic"


class CueType(StrEnum):
    """What kind of starting point a cue represents.

    - ``SYMBOL``: a function/class/variable name.
    - ``CONCEPT``: an LLM-extracted phrase (e.g. "rate limiting").
    - ``ERROR_MESSAGE``: a literal error/exception string.
    """

    SYMBOL = "symbol"
    CONCEPT = "concept"
    ERROR_MESSAGE = "error_message"


class TagCategory(StrEnum):
    """Closed set of semantic-bridge categories a tag may belong to."""

    MODULE_PATH = "module_path"
    ARCH_LAYER = "arch_layer"
    PATTERN_TYPE = "pattern_type"
    LANG_CONSTRUCT = "lang_construct"
    LANGUAGE = "language"


class ContentKind(StrEnum):
    """The concrete artifact a content node captures."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    COMMIT = "commit"
    TEST = "test"


class EdgeType(StrEnum):
    """Named, directional relationships between nodes."""

    CUE_OF = "cue_of"
    TAGGED_WITH = "tagged_with"
    PART_OF_TOPIC = "part_of_topic"
    REDIRECTS_TO = "redirects_to"


class Direction(StrEnum):
    """Direction of edge traversal relative to a node."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
