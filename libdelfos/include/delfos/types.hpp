#pragma once

#include <cstdint>
#include <limits>

namespace delfos {

using NodeIdx = std::uint32_t;
using EdgeIdx = std::uint32_t;

inline constexpr NodeIdx INVALID_NODE = std::numeric_limits<NodeIdx>::max();
inline constexpr EdgeIdx INVALID_EDGE = std::numeric_limits<EdgeIdx>::max();

enum class NodeType : std::uint8_t { Cue, Tag, Content };
enum class NodeStatus : std::uint8_t { Active, Deleted };
enum class EdgeType : std::uint8_t { CueOf, TaggedWith, PartOfTopic, RedirectsTo };
enum class CueType : std::uint8_t { Symbol, Concept, ErrorMessage };
enum class TagCategory : std::uint8_t { ModulePath, ArchLayer, PatternType, LangConstruct, Language };
enum class ContentKind : std::uint8_t { Function, Class, Module, Commit, Test };
enum class MemoryLayer : std::uint8_t { Episodic, Semantic, Topic };
enum class Direction : std::uint8_t { Outgoing, Incoming };

}  // namespace delfos
