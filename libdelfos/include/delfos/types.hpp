#pragma once
#include <cstdint>
#include <limits>

namespace delfos {

using NodeIdx = uint32_t;
using EdgeIdx = uint32_t;

inline constexpr NodeIdx INVALID_NODE = std::numeric_limits<NodeIdx>::max();
inline constexpr EdgeIdx INVALID_EDGE = std::numeric_limits<EdgeIdx>::max();

enum class NodeType : uint8_t { Cue, Tag, Content };
enum class NodeStatus : uint8_t { Active, Deleted };
enum class EdgeType : uint8_t { CueOf, TaggedWith, PartOfTopic, RedirectsTo };
enum class CueType : uint8_t { Symbol, Concept, ErrorMessage };
enum class TagCategory : uint8_t { ModulePath, ArchLayer, PatternType, LangConstruct, Language };
enum class ContentKind : uint8_t { Function, Class, Module, Commit, Test };
enum class MemoryLayer : uint8_t { Episodic, Semantic, Topic };
enum class Direction : uint8_t { Outgoing, Incoming };

} // namespace delfos
