Title: Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents

URL Source: https://arxiv.org/html/2606.06036

Published Time: Fri, 05 Jun 2026 00:52:21 GMT

Markdown Content:
###### Abstract

Despite recent progress, LLM agents still struggle with reasoning over long interaction histories. While current memory-augmented agents rely on a static “retrieve-then-reason” paradigm, this rigid pipeline design prevents them from dynamically adapting memory access to intermediate evidence discovered during inference. To bridge this gap, we propose MRAgent, a framework that combines an associative memory graph with an active reconstruction mechanism. We represent memory as a Cue–Tag–Content graph, where associative tags serve as semantic bridges connecting fine-grained cues to memory contents. Operating on this structure, our active reconstruction mechanism integrates LLM reasoning directly into memory access, allowing the agent to iteratively explore and prune retrieval paths based on accumulated evidence. This ensures that memory retrieval is dynamically adapted to the reasoning context while avoiding combinatorial explosion caused by unconstrained expansion. Experiments on the LoCoMo benchmark and LongMemEval benchmark demonstrate significant improvements over strong baselines (up to 23\%), while substantially reducing token and runtime cost, highlighting the effectiveness of active and associative reconstruction for long-horizon memory reasoning. The code is available at: [https://github.com/Ji-shuo/MRAgent](https://github.com/Ji-shuo/MRAgent).

LLM Agents, Memory System

## 1 Introduction

LLMs exhibit a “jagged” cognitive profile (Hendrycks et al., [2025](https://arxiv.org/html/2606.06036#bib.bib8)), excelling at math and reasoning, but deficient in tasks requiring long-term memory, such as interactive assistance or decision-support systems acting over extended interactions(Gao et al., [2026](https://arxiv.org/html/2606.06036#bib.bib5)). In such long-horizon tasks, LLMs are fundamentally constrained by their limited context windows, which restrict their ability to retain interaction history over time (Hatalis et al., [2023](https://arxiv.org/html/2606.06036#bib.bib7)).

To mitigate these limitations, prior work equips LLM agents with external memory systems. Early approaches adopt Retrieval-Augmented Generation (RAG)(Lewis et al., [2020](https://arxiv.org/html/2606.06036#bib.bib14)), where memory access is realized via similarity-based retrieval over unstructured text or embedding stores. Subsequent work introduces more structured memory representations, including hierarchical stores(Fang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib3); Kang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib12)) and knowledge graphs (KGs)(Rasmussen et al., [2025](https://arxiv.org/html/2606.06036#bib.bib20); Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24); Huang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib10)), which explicitly encode entities and relations to support more interpretable and relational retrieval. However, retrieval in these systems remains restricted to fixed top-k selection or predefined subgraph traversal, failing to infer new retrieval cues or adapt to intermediate evidence. Under this formulation, existing memory systems operate as _passive retrieval policies_.

![Image 1: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_intro2.png)

Figure 1: Comparison between passive retrieval and active memory reconstruction in MRAgent.

![Image 2: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_motivation.png)

Figure 2:  Illustrative example comparing passive retrieval and active reconstruction: passive retrieval only retrieves memory content related to Nate’s video game tournaments based on the query, while active reconstruction infers a critical temporal cue (“July”) through LLM reasoning and identifies Caroline’s corresponding activity. 

In contrast, cognitive neuroscience conceptualizes memory retrieval as an _active_ and _associative_ reconstruction process(Rugg & Renoult, [2025](https://arxiv.org/html/2606.06036#bib.bib21)), rather than a passive read-out of stored content. Specifically, retrieval is initiated by contextual cues, which propagate through intermediate representations, progressively reconstructing coherent memory experiences(Frankland & Josselyn, [2019](https://arxiv.org/html/2606.06036#bib.bib4)).

This perspective highlights two key challenges for LLM-based memory systems, as illustrated in Figure[1](https://arxiv.org/html/2606.06036#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"). (Challenge 1: Active Reconstruction) How can memory access be transformed from one-shot retrieval into an active multi-step reconstruction process that progressively reveals information across reasoning steps? (Challenge 2: Associative Memory Structure) How can memory be organized to capture semantic and structural dependencies, enabling guided exploration across associative items?

In this light, we propose the M emory R easoning Architecture for LLM Agent s (MRAgent), a framework that enables LLM agents to perform active and associative memory reconstruction by integrating LLM reasoning into memory access. MRAgent explores multiple candidate retrieval paths over a memory graph, pruning irrelevant branches based on intermediate evidence, and iteratively optimizing for the next step with the highest information gain. To enable controlled memory traversal, we design a Cue–Tag–Content memory graph, where tags encode associations between fine-grained cues and specific memory contents. By exposing these associations explicitly, tags allow the LLM to select retrieval paths before accessing detailed memory contents. Our contributions can be summarized as follows:

*   •
We propose active memory reconstruction, a new paradigm that integrates memory access into the reasoning process, allowing the agent to dynamically adapt search strategy based on intermediate evidence.

*   •
We introduce a Cue–Tag–Content memory graph in which associative tags mediate retrieval between cues and content, enabling the LLM to identify promising retrieval paths while pruning irrelevant branches.

*   •
We provide a theoretical analysis proving that active retrieval policies are strictly more expressive than passive retrieval.

*   •
Extensive experiments demonstrate that MRAgent outperforms strong baselines with significantly improved token and runtime efficiency.

## 2 Problem Setting: Active Memory Access

In this section, we formulate memory retrieval as a sequential decision process and analyze why the traditional passive retrieval paradigm fails on complex, multi-step queries.

### 2.1 Memory Retrieval Definition

We consider an external memory \mathcal{M} consisting of memory units \mathcal{V}=\{v_{1},\dots,v_{N}\}. Given a query x, memory access proceeds by sequentially selecting memory units over T steps. Let S^{(t)}=\{v^{(1)},\ldots,v^{(t)}\} denote the accumulated evidence after step t. We distinguish two classes of memory access policies.

Passive retrieval. A _passive_ (stateless) retrieval policy \pi_{\mathrm{p}} selects memory units solely based on the query:

\{v^{(1)},\ldots,v^{(T)}\}=\pi_{\mathrm{p}}(x).(1)

Active reconstruction. An _active_ (stateful) policy \pi_{\mathrm{a}}^{(t)} selects the next unit conditioned on the evolving evidence:

v^{(t)}=\pi_{\mathrm{a}}^{(t)}(x,S^{(t-1)}),\qquad S^{(t)}=S^{(t-1)}\cup\{v^{(t)}\}.(2)

This formulation enables adaptive, multi-step interaction with memory.

### 2.2 Limits of Passive Retrieval

We categorize retrieval strategies of existing memory systems into two paradigms based on their memory organization and adaptation mechanisms, and compare them with the proposed active reconstruction paradigm in Figure[2](https://arxiv.org/html/2606.06036#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

Similarity-based Retrieval. Many existing memory systems adopt similarity-based retrieval, such as MemoryBank(Zhong et al., [2024](https://arxiv.org/html/2606.06036#bib.bib25)) and Mem0(Chhikara et al., [2025](https://arxiv.org/html/2606.06036#bib.bib2)), where retrieval is defined by a similarity scoring function:

\pi_{\text{sim}}(x)=\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},k\right).(3)

In this paradigm, memory serves as a static context provider and the relevance function \mathrm{sim}(x,v) is fixed by the query. As shown in Figure[2](https://arxiv.org/html/2606.06036#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")(a), such methods retrieve many events related to “video game tournament” based on surface-level relevance, thereby introducing substantial noise while failing to find the correct evidence.

Graph-based Memory Retrieval. To exploit structural relations among memory units, methods such as A-Mem(Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24)), Zep(Rasmussen et al., [2025](https://arxiv.org/html/2606.06036#bib.bib20)) introduce graph-structured memory. These approaches extend the retrieval mechanism by combining similarity-based seeding with predefined N-hop neighbor expansion:

\displaystyle\mathcal{V}^{\text{sim}}\displaystyle=\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},k\right),(4)
\displaystyle\pi_{\text{graph}}(x)\displaystyle=\mathcal{V}^{\text{sim}}\cup\mathrm{Neighbor}\!\left(\mathcal{V}^{\text{sim}}\right).

where \mathrm{Neighbor}(\cdot) returns the predefined N-hop neighbors of a node set in the memory graph. While this strategy alleviates certain multi-hop retrieval challenges, it requires relevant evidence to be connected through explicit graph links. Moreover, the use of fixed neighbor expansion often introduces substantial noise. Figure[2](https://arxiv.org/html/2606.06036#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")(b) shows that graph-based expansion retrieves additional but irrelevant neighbors, yet still fails to recover information about Caroline, which is not directly connected to “video game tournament” events in the memory graph.

Active Memory Reconstruction. By integrating LLM reasoning directly into memory access, active reconstruction enables the agent to infer new retrieval cues and adapt traversal strategies based on accumulated evidence. As illustrated in Figure[2](https://arxiv.org/html/2606.06036#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")(c), intermediate findings can be explicitly transformed into new retrieval constraints, allowing the agent to recover evidence that is unreachable under passive policies.

Summary. The fundamental limitation of passive retrieval lies in its inability to perform reasoning concurrently with memory access, leading to three weaknesses: (i) an inability to revise strategies based on intermediate state, such as identifying ‘July’ as a temporal anchor; (ii) the accumulation of noise due to fixed aggregation ; and (iii) heavy reliance on pre-constructed structures, limiting flexibility and scalability. A detailed analysis is included in Appendix[A](https://arxiv.org/html/2606.06036#A1 "Appendix A Detailed Analysis of Passive Retrieval ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

### 2.3 Motivation from Cognitive Memory Systems

![Image 3: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_relatedwork2.png)

Figure 3:  Functional correspondence between human memory reconstruction and the MRAgent architecture. 

These limitations highlight the necessity of moving beyond static retrieval toward an active memory reconstruction paradigm, which is strongly supported by cognitive neuroscience research(Rugg & Renoult, [2025](https://arxiv.org/html/2606.06036#bib.bib21)). As illustrated in Figure[3](https://arxiv.org/html/2606.06036#S2.F3 "Figure 3 ‣ 2.3 Motivation from Cognitive Memory Systems ‣ 2 Problem Setting: Active Memory Access ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), studies in human memory suggest that recall unfolds sequentially: contextual cues trigger the reactivation of _engrams_, compact internal memory states formed from past experiences, whose activation in turn biases and constrains subsequent recall, progressively reconstructing a coherent memory(Rashid et al., [2016](https://arxiv.org/html/2606.06036#bib.bib19)). Motivated by this perspective, we adopt a Cue–Tag–Content architecture, where tags serve as intermediate associative structures that encode how cues are linked to memory contents and guide the memory reconstruction process. Memory content can be distinguished into episodic memory for concrete events and semantic memory for shared concepts and knowledge, as supported by cognitive neuroscience(Manns et al., [2003](https://arxiv.org/html/2606.06036#bib.bib17)).

## 3 Associative Memory System

![Image 4: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_framework.png)

Figure 4: MRAgent: An Associative Memory System with LLM-Driven Active Memory Reconstruction. (a) MRAgent constructs an associative memory system from dialogues, organizing episodic and semantic memories through Cue–Tag–Content structures that explicitly encode semantic and relational associations through tags. (b) Upon a query, the agent performs active memory reconstruction, where an LLM iteratively reasons over cues and tags to traverse the memory graph and selectively reconstruct task-relevant memory.

To enable the active reconstruction paradigm described in Section[2](https://arxiv.org/html/2606.06036#S2 "2 Problem Setting: Active Memory Access ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), we organize the agent’s memory as a heterogeneous graph. This section details the core _Cue–Tag–Content_ architecture and its organization into multi-granular layers.

### 3.1 Cue–Tag–Content Associative Memory

Motivated by the need for active memory reconstruction in Section[2](https://arxiv.org/html/2606.06036#S2 "2 Problem Setting: Active Memory Access ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), we organize memory as a structured associative graph rather than a flat collection of retrievable items.

As illustrated in Figure[4](https://arxiv.org/html/2606.06036#S3.F4 "Figure 4 ‣ 3 Associative Memory System ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") (a), the memory construction pipeline consists of two phases: element generation and graph construction. The first phase employs LLM to extract and generate memory elements from dialogs. Then, these elements are utilized to construct a memory graph.

Unified Memory System. We model the memory system as a heterogeneous graph \mathcal{M}=(\mathcal{C},\mathcal{V},\mathcal{R}). The graph contains two primary categories of nodes: (1) Cues c\in\mathcal{C} are fine-grained keywords such as entities or attributes. (2) Contents v\in\mathcal{V} store specific memory items.

Associations between cues and contents are encoded as typed relations: \mathcal{R}\subseteq\mathcal{C}\times\mathcal{G}\times\mathcal{V}, where each triple (c,g,v)\in\mathcal{R} connects a cue c to a content node v through a relation attribute g, referred to as a Tag.

Associative Tags. Tags are utilized to summarize associative relations between fine-grained cues and content units. Based on these associations, the LLM performs a _two-stage retrieval_ process, where it first selects a small set of relevant tags and then retrieves content conditioned on the selected tags. In large and complex memory graphs, naively expanding n-hop neighbors from content nodes often leads to combinatorial explosion and the inclusion of substantial irrelevant memories. By introducing tags as explicit associative intermediates, MRAgent enables guided and flexible reasoning over the memory graph. Tags provide semantic guidance, allowing the agent to evaluate and prune traversal branches to avoid incurring the computational cost of processing full episodic content.

To realize this two-stage retrieval process, we define two induced mapping operators over the memory relation \mathcal{R}. Specifically, \phi_{c\rightarrow g} activates candidate associative tags from a given cue, while \phi_{(c,g)\rightarrow v} retrieves content conditioned on both the cue and a selected tag:

\displaystyle\phi_{c\rightarrow g}(c)\displaystyle\triangleq\{g\mid(c,g,\cdot)\in\mathcal{R}\},(5)
\displaystyle\phi_{(c,g)\rightarrow v}(c,g)\displaystyle\triangleq\{v\mid(c,g,v)\in\mathcal{R}\}.

Together, these operators decouple associative reasoning from content-level retrieval, making selective and reconstructive memory access tractable in large-scale graphs.

### 3.2 Multi-Granular Memory Layers

Inspired by human memory, we organize memory contents into complementary types spanning concrete events, stable knowledge, and higher-level abstractions. Episodic memory preserves event-specific information grounded in particular contexts, semantic memory captures abstract and relatively stable knowledge distilled across events, and topic-level abstractions summarize recurring patterns shared across episodes. Based on this distinction, we organize the memory graph into multiple functional layers, each supporting different forms of reasoning and retrieval.

Episodic Layer (Cue–Tag–Episode). The episodic layer stores event-specific memory units e_{i}\in\mathcal{V}^{e}, each corresponding to a concrete experience occurring at a particular time. Episodes are retrievable through a set of fine-grained cues \mathcal{C}_{i}\subseteq\mathcal{C} such as entities, actions, or contextual keywords and are routed by tags g_{i} that summarize associative relations between cues and episodic content. To support temporal reasoning, episodic memories are further organized along a unified timeline, allowing temporal constraints to be imposed during reconstruction.

Semantic Layer (Cue–Tag–Semantic). The semantic layer captures relatively stable knowledge s_{i}\in\mathcal{V}^{s}, such as personal attributes, preferences, and general facts extracted from raw dialogue content. Each semantic node is anchored to a cue c_{i}, while the tag encode aspect-level associations of that cue, such as personality traits, long-term preferences, or factual attributes. This design enables direct access to targeted semantic information without requiring retrieval over potentially long episodic histories.

Abstraction Layer (Topics). The abstraction layer stores topic nodes \tau\in\mathcal{V}^{\tau}, each summarizing recurring patterns shared across a coherent set of episodes. Topics are connected to their constituent episodes and serve as a computational abstraction that supports efficient top-down transitions \phi_{\tau\to e}, allowing the agent to first localize a relevant topic and then descend to the associated episodes.

Together, these multi-granular memory layers allow MRAgent to flexibly combine event-level evidence from episodic memory, abstract knowledge from semantic memory, and higher-level structure from topic abstractions, supporting both contextualized and conceptual reasoning.

### 3.3 Memory Population via LLM Distillation

The memory system is populated through an automated distillation pipeline applied to the input stream T. The stream is first segmented into episodic units e_{i}, each corresponding to a coherent event grounded in a specific context. For every episodic unit, tags and cues are extracted using LLM-based components:

g_{i}=F_{\text{LLM}}^{\text{tag}}(e_{i}),\qquad C_{i}=F_{\text{LLM}}^{\text{cue}}(e_{i}),(6)

where F_{\text{LLM}}^{\text{tag}} produces a short associative tag g_{i} that summarizes the relational pattern of the episode, and F_{\text{LLM}}^{\text{cue}} extracts a set of fine-grained cues C_{i} such as entities, attributes, and salient descriptors. Each cue in C_{i} is then linked to the episodic unit e_{i} through the tag g_{i}, forming the Cue–Tag–Episode relations that constitute the episodic layer of the memory graph. Semantic units are extracted in a similar manner, yielding stable, abstract knowledge that persists across individual episodes and is anchored to entity-level cues through aspect-level tags.

To support reasoning at a coarser granularity, topic nodes are generated by summarizing the shared themes of related episodes, and each topic node is connected to its constituent episodes. This layered distillation organizes the input stream into a multi-granular associative structure, allowing the agent to access memory at the level of concrete events, stable facts, or higher-level topics as required during reconstruction. Additional details are provided in Appendix[B.1](https://arxiv.org/html/2606.06036#A2.SS1 "B.1 Memory Construction Pipeline ‣ Appendix B Implementation and Execution Details ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

## 4 MRAgent: Reconstructive Memory Agent

In this section, we present MRAgent, which formulates memory access as an active reconstruction process. We first define the reconstruction state and traversal actions (Section[4.1](https://arxiv.org/html/2606.06036#S4.SS1 "4.1 Reconstructive Memory Framework ‣ 4 MRAgent: Reconstructive Memory Agent ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")), and then describe the memory reconstruction process in detail (Section[4.2](https://arxiv.org/html/2606.06036#S4.SS2 "4.2 Memory Reconstruction Process ‣ 4 MRAgent: Reconstructive Memory Agent ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")). Finally, we provide a theoretical analysis of its expressivity (Section[4.3](https://arxiv.org/html/2606.06036#S4.SS3 "4.3 Theoretical Analysis: Active vs. Passive Retrieval ‣ 4 MRAgent: Reconstructive Memory Agent ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")).

### 4.1 Reconstructive Memory Framework

Rather than executing a predefined retrieval pipeline, MRAgent performs active reconstruction within a structured memory system. This process is formalized through an explicit reconstruction state and a set of traversal actions, which collectively characterize possible reconstruction trajectories.

#### Reconstruction State.

MRAgent maintains an explicit reconstruction state that guides the selection of subsequent traversal directions during memory reconstruction. At step t, the reconstruction state is defined as

\mathcal{S}^{(t)}=(\mathcal{Z}^{(t)},\mathcal{H}^{(t)}),(7)

where \mathcal{Z}^{(t)} denotes the _active set_ of memory elements (including cues, tags and contents), serving as the candidates for the next traversal step, and \mathcal{H}^{(t)} denotes the _reconstructed context_ consisting of evidence accumulated in previous steps, which conditions subsequent traversal directions.

#### Traversal Actions.

We define a finite set of traversal actions \mathcal{A}=\{\Pi_{1},\ldots,\Pi_{m}\} to specify how the LLM explores the structured memory graph during reconstruction. Each traversal action is induced by a predefined mapping operator \phi introduced in Eq.([5](https://arxiv.org/html/2606.06036#S3.E5 "Equation 5 ‣ 3.1 Cue–Tag–Content Associative Memory ‣ 3 Associative Memory System ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")), including Cue\rightarrow Tag, (Cue,Tag)\rightarrow Content, and Content\rightarrow(Cue,Tag).

_Forward traversal_ actions expands the active set along Cue–Tag–Content relations, enabling the agent to retrieve new memory contents conditioned on the current cues and tags. Specifically, \Pi_{c\rightarrow g} activates associative tags from a given set of cues \mathcal{C}^{(t)}, and \Pi_{(c,g)\rightarrow v} retrieves memory content jointly conditioned on selected cues and tags \big(\mathcal{C}^{(t)},\mathcal{G}^{(t)}\big):

\displaystyle\Pi_{c\rightarrow g}\!\big(\mathcal{C}^{(t)}\big)\displaystyle\triangleq\bigcup_{c^{\prime}\in\mathcal{C}^{(t)}}\phi_{c\rightarrow g}(c^{\prime}),(8)
\displaystyle\Pi_{(c,g)\rightarrow v}\!\big(\mathcal{C}^{(t)},\mathcal{G}^{(t)}\big)\displaystyle\triangleq\bigcup_{c^{\prime}\in\mathcal{C}^{(t)}}\ \bigcup_{g^{\prime}\in\mathcal{G}^{(t)}}\phi_{(c,g)\rightarrow v}(c^{\prime},g^{\prime}).

_Reverse traversal_ actions allows retrieved content to activate new cues and tags for subsequent exploration, enabling the agent to refine or redirect its reconstruction trajectory based on intermediate evidence. Formally, \Pi_{v\rightarrow(c,g)} identifies cues and tags associated with the retrieved content:

\Pi_{v\rightarrow(c,g)}\!\big(\mathcal{V}^{(t)}\big)\triangleq\{(c^{\prime},g^{\prime})\mid\exists v^{\prime}\in\mathcal{V}^{(t)},\ (c^{\prime},g^{\prime},v^{\prime})\in\mathcal{R}\},(9)

Together, these forward and reverse traversal actions define a compositional action space that allows the LLM to expand and refine memory traversal during reconstruction.

### 4.2 Memory Reconstruction Process

With the reconstruction state and traversal actions defined, we detail the memory reconstruction process, which iteratively explores relevant memory contents through state updates and traversal actions. Given a query, MRAgent first extracts a set of fine-grained cues and matches them against the stored cue set, obtaining the initial active set \mathcal{Z}^{(0)} and the initial reconstruction state \mathcal{S}^{(0)}=(\mathcal{Z}^{(0)},\varnothing). Starting from this state, MRAgent starts an iterative reconstruction loop consisting of LLM reasoning, controlled memory traversal, and LLM-guided routing.

#### LLM Reasoning and Action Selection.

Conditioned on reconstruction state, the LLM reasons over the query x, the accumulated context \mathcal{H}^{(t)} to select a set of traversal actions \mathcal{A}^{(t)}\subseteq\mathcal{A} based on current active set \mathcal{Z}^{(t)}:

\mathcal{A}^{(t)}=f_{\text{select}}\!\big(x,\mathcal{H}^{(t)},\mathcal{Z}^{(t)}\big),(10)

where f_{\text{select}} denotes the LLM-based action-selection function that selects promising directions for expansion, thereby reducing noise. Furthermore, conditioning on the accumulated evidence \mathcal{H}^{(t)} enables the agent to discover new cues and dynamically adjust its reasoning trajectory.

#### Controlled Memory Traversal.

For each selected traversal action \Pi_{a}\in\mathcal{A}^{(t)}, the memory system executes the corresponding traversal operator to generate candidate nodes:

\widetilde{\mathcal{Z}}^{(t+1)}=\bigcup_{a\in\mathcal{A}^{(t)}}\Pi_{a}\!\big(\mathcal{Z}^{(t)}\big),(11)

where \widetilde{\mathcal{Z}}^{(t+1)} is the new candidate set. This step expands the traversal trajectories, guided by LLM-selected actions rather than exhaustive graph expansion.

#### LLM Routing and State Update.

Based on the generated candidates, the LLM selects the most relevant content and prunes irrelevant branches to ensure the accumulated context remains concise and focused. Formally, the next active set and reconstruction state are updated as follows:

\displaystyle\mathcal{Z}^{(t+1)}\displaystyle=f_{\text{route}}\!\big(x,\mathcal{H}^{(t)},\widetilde{\mathcal{Z}}^{(t+1)}\big),(12)
\displaystyle\mathcal{H}^{(t+1)}\displaystyle=\mathcal{H}^{(t)}\cup\mathcal{Z}^{(t+1)},

where f_{\text{route}} is the LLM-based routing function that evaluates semantic associations between the query, the reconstructed context, and newly retrieved memory contents. Unlike surface-level matching or predefined traversal, this routing step incorporates semantic associations and structural relations exposed by the memory graph. Following the state update, the accumulated context \mathcal{H}^{(t+1)} is evaluated by \mathcal{C}_{\text{LLM}} to determine if the evidence is sufficient to answer the query or if further exploration is required.

Through this process, MRAgent integrates LLM reasoning directly into the multi-turn memory reconstruction. Selective evidence expansion reduces noise and improves efficiency, while conditioning on intermediate evidence enables flexible adjustment of the reasoning trajectory. Implementation details and algorithms are provided in Appendix[B.2](https://arxiv.org/html/2606.06036#A2.SS2 "B.2 Executable Memory Reconstruction ‣ Appendix B Implementation and Execution Details ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

### 4.3 Theoretical Analysis: Active vs. Passive Retrieval

MRAgent is designed as an _active_ retrieval approach over a graph-based memory. In this section, we formalize the advantage of such active retrieval over passive retrieval, from an approximation-theoretic view. For generality and notational simplicity, our theory generalizes from the (cue, tag, episode) formulation in MRAgent to arbitrary heterogeneous graphs with textual information on each node.

Given retrieval budget T, recall that _active reconstruction_ adaptively chooses the next node to retrieve based on what it has already retrieved, while a _passive_ retriever must commit to all T retrievals upfront as a function of the query alone.

These retrieval strategies induce different _hypothesis classes_ (sets of input–output mappings). Specifically, let \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T) contain all predictors implementable by an LM that can make T adaptive retrieval calls to the graph, and let \mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T) contain those implementable when the T retrieval calls are fixed in advance (non-adaptive). Then our main result is:

###### Theorem 4.1(Active retrieval is strictly more powerful than passive retrieval).

For any retrieval budget T\geq 2, the passive hypothesis class is strictly contained in the active hypothesis class:

\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T)\ \subsetneq\ \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T).

Intuitively, LMs with active retrieval can learn any function that LMs with passive retrieval can, but not vice versa. The full statement and proofs are given in Appendix[C](https://arxiv.org/html/2606.06036#A3 "Appendix C Theoretical Analysis: Active vs. Passive Retrieval over a Heterogeneous KG Memory ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

## 5 Experiments

Table 1: Performance across different question types on LoCoMo. Evaluation metrics include F1 score (F1), and LLM-Judge score (J). All values are reported as percentages. Bold indicates the best result, and underline indicates the second best.

In this section, we conduct comprehensive experiments to evaluate MRAgent and answer the following research questions: (RQ1) How does MRAgent perform compared to existing memory systems? (RQ2) How does MRAgent compare to baselines in terms of computational cost? (RQ3) How do different components of MRAgent contribute to overall performance? (RQ4) How does multi-turn reasoning progressively improve memory reconstruction? (RQ5) How does MRAgent behave in real conversational scenarios?

### 5.1 Experiment Setup

Benchmarks. We evaluate MRAgent on two widely used benchmarks for long-context memory evaluation: (1) _LoCoMo_(Maharana et al., [2024](https://arxiv.org/html/2606.06036#bib.bib16)), which focuses on long conversational memory understanding, and (2) _LongMemEval_(Wu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib23)), which is designed to assess long-term memory system across multiple sessions with longer interaction histories per query.

Baselines. We evaluate MRAgent against representative memory-augmented baselines: Retrieval-augmented generation (RAG), _LangMem_(LangChain, [2025](https://arxiv.org/html/2606.06036#bib.bib13)), _A-Mem_(Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24)), _MemoryOS_(Kang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib12)), and _Mem0_(Chhikara et al., [2025](https://arxiv.org/html/2606.06036#bib.bib2)).

Implementation. We evaluate all methods using two LLM backbones: Gemini-2.5-Flash and Claude-Sonnet-4.5. Following prior work, we report F 1 and LLM-Judge (J) scores using GPT-4o-mini, and additionally report evidence recall (Recall) for analysis.

Additional details are reported in the Appendix[D](https://arxiv.org/html/2606.06036#A4 "Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

### 5.2 Main Results (RQ1)

Table 2: Performance on LongMemEval evaluated by LLM-Judge\uparrow. Best and second-best results among Gemini-backbone methods are marked in bold and underline, respectively. Mul.: multi-session; Sgl.: single-session-user; Tmp.: temporal-reasoning; Pref.: single-session-preference. MRAgent∗ uses Claude for retrieval while memories are constructed by Gemini. The complete table is provided in Appendix[D.5](https://arxiv.org/html/2606.06036#A4.SS5 "D.5 Detailed Results on LongMemEval ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

MRAgent consistently outperforms all baselines across question types and backbones. As shown in Table[1](https://arxiv.org/html/2606.06036#S5.T1 "Table 1 ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), MRAgent improves the overall J score from 68.31 to 84.21 under the Gemini backbone (an 23.3\% relative gain), and achieves a 12.4\% improvement under the Claude backbone.

These gains can be attributed to two key factors. First, the Cue–Tag–Content (CTC) memory design explicitly encodes associative relations via tags, integrating semantic information into the memory graph structure. Compared with graph-based retrieval methods that expand purely along predefined structural links, the CTC memory allows MRAgent to select retrieval directions based on semantic relevance, thereby improving its ability to locate supporting evidence. Second, MRAgent integrates LLM reasoning into multi-turn memory access, allowing retrieval directions to adapt to accumulated evidence and to progressively form coherent reasoning chains. This adaptive reconstruction process enables the agent to focus on relevant evidence and follow evidence-guided retrieval paths, leading to superior performance over passive retrieval baselines.

Table[2](https://arxiv.org/html/2606.06036#S5.T2 "Table 2 ‣ 5.2 Main Results (RQ1) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") further shows consistent improvements on LongMemEval, achieving a relative improvement of 32\% over the strongest baseline.

### 5.3 Cost Analysis (RQ2)

Table 3: Per-sample token consumption and runtime on long-term LongMemEval across different memory systems using the Gemini backbone. Results include both memory construction and retrieval. Bold indicates the best result, and underline indicates the second best.

Table LABEL:tab:token-consumption reports the total token consumption and runtime per sample on LongMemEval, which is designed for long-term memory assess across multiple dialogue sessions.

MRAgent improves information efficiency through selective, on-demand memory access. MRAgent reduces prompt tokens to 118k, a significant decrease from baselines like A-Mem (632k). Unlike existing methods that repeatedly summarize history and analyze intricate dependencies during construction, MRAgent maintains a lightweight construction phase and defers complex relation-building to the retrieval stage, where it is performed in a query-specific manner. Moreover, by leveraging associative tags to semantically guide retrieval directions, the agent can prune irrelevant paths before accessing expensive episodic content. This “on-demand” approach ensures that computational resources are focused strictly on query-relevant evidence.

### 5.4 Ablation Study (RQ3)

![Image 5: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_ablation_study.png)

Figure 5: Ablation results on LoCoMo for multi-hop queries, evaluated using Recall and LLM-Judge (J) under Claude backbone.

Figure[5](https://arxiv.org/html/2606.06036#S5.F5 "Figure 5 ‣ 5.4 Ablation Study (RQ3) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") presents ablation results on multi-hop questions in LoCoMo, isolating the contributions of memory structure and reasoning mechanisms. We consider three structural variants: CE (Cue\rightarrow Episode) with direct indexing, CTE (Cue–Tag–Episode) with mediated episodic retrieval, and CTC (Cue–Tag–Content) with the full memory structure. The CTE and CTC variant is evaluated both without reasoning (green bars) and with reasoning (blue bars).

Active multi-step reasoning is a primary factor underlying the observed performance gains. Across all memory structures, variants with reasoning (blue bars) consistently outperform their structure-only counterparts (green bars). This indicates that multi-step reasoning and traversal are crucial for accumulating evidence and supporting multi-hop inference, whereas one-shot retrieval is insufficient to resolve complex multi-hop queries.

Associative tags provide effective semantic guidance for retrieval. Within the no-reasoning setting (green bars), performance improves monotonically from CE to CTE to CTC, demonstrating that richer associative structures enable more reliable retrieval. In particular, tags help guide retrieval toward semantically relevant directions and reduce the inclusion of fragmented or irrelevant memory units.

Episodic and semantic memory layers are complementary. Removing the semantic memory component leads to a clear performance degradation (blue bars). While episodic memory preserves event-specific details, semantic memory captures stable, abstract knowledge that is essential for multi-hop reasoning.

### 5.5 Multi-turn Reasoning Analysis (RQ4)

![Image 6: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_multi_turns.png)

Figure 6: Analysis of multi-turn reasoning on LoCoMo under the Claude backbone.

As shown in Figure[6](https://arxiv.org/html/2606.06036#S5.F6 "Figure 6 ‣ 5.5 Multi-turn Reasoning Analysis (RQ4) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") (a), we evaluate cumulative evidence recall across reasoning turns on the LoCoMo dataset. Figure[6](https://arxiv.org/html/2606.06036#S5.F6 "Figure 6 ‣ 5.5 Multi-turn Reasoning Analysis (RQ4) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") (b) summarizes the average number of reasoning turns to convergence (Average Turns) and the maximum number of turns that retrieve valid information (Max Valid Turns).

Multi-turn reasoning progressively recovers missing evidence. As shown in Figure[6](https://arxiv.org/html/2606.06036#S5.F6 "Figure 6 ‣ 5.5 Multi-turn Reasoning Analysis (RQ4) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") (a), single-hop (blue line) and temporal (yellow line) queries reach near-perfect recall within approximately three turns, whereas multi-hop queries benefit substantially from iterative exploration, with recall improving by over 30\% across successive steps (red line). This highlights the necessity of multi-step reconstruction for resolving compositional and long-range dependencies.

The agent autonomously evaluates accumulated context to guide search and termination. As shown in Figure[6](https://arxiv.org/html/2606.06036#S5.F6 "Figure 6 ‣ 5.5 Multi-turn Reasoning Analysis (RQ4) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") (b), Max Valid Turns closely matches Average Turns, suggesting that the LLM effectively determines when to continue searching and when to stop. This behavior minimizes redundant exploration. Furthermore, experiments in Appendix[9](https://arxiv.org/html/2606.06036#A4.F9 "Figure 9 ‣ D.6 Budget Sensitivity Analysis of Multi-step Reconstruction ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") show that increasing the parallel retrieval budget cannot substitute for deeper reconstruction depth.

### 5.6 Case Study (RQ5)

![Image 7: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig-casestudy.png)

Figure 7:  Reasoning trajectory of MRAgent over the memory graph for a multi-session query. Starting from the cue “Jonna”, the agent traverses multiple associative tags to retrieve both episodic memories (e.g., screenplay submissions) and semantic information (e.g., background about the screenplay). It then reasons over higher-level topics to recover rejection events and aligns them with previously retrieved submissions, enabling a coherent answer to the query. Here, Dx\!:\!x denotes concrete event instances. 

We include a qualitative case study in Figure[7](https://arxiv.org/html/2606.06036#S5.F7 "Figure 7 ‣ 5.6 Case Study (RQ5) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") demonstrating MRAgent’s ability to actively reconstruct multi-session evidence over a structured memory graph for complex, temporally grounded queries, with detailed analysis provided in the Appendix[D.8](https://arxiv.org/html/2606.06036#A4.SS8 "D.8 Case Studies ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents").

## 6 Related Work

Retrieval-Augmented Generation. RAG(Lewis et al., [2020](https://arxiv.org/html/2606.06036#bib.bib14)) augments LLMs by injecting external documents into prompts via similarity search, where memory is an unstructured vector store and retrieval reduces to a one-shot query-time top-k selection. Two variants extend this paradigm. GraphRAG(Han et al., [2025](https://arxiv.org/html/2606.06036#bib.bib6)) organizes the retrieval corpus into graph structures and retrieves through community summaries and neighborhood expansion, improving global and multi-hop reasoning over flat similarity search. Agentic RAG instead moves retrieval inside the reasoning loop. Search-o1(Li et al., [2025](https://arxiv.org/html/2606.06036#bib.bib15)) couples large reasoning models with an agentic mechanism that issues search queries on demand whenever a knowledge gap is detected and refines retrieved documents before integrating them into the reasoning chain, and Search-R1(Jin et al., [2025](https://arxiv.org/html/2606.06036#bib.bib11)) trains this behavior through reinforcement learning over multi-turn query generation. These methods retrieve from open or external corpora to fill factual knowledge gaps within a single reasoning task, rather than reconstructing over an agent’s own persistent interaction history.

Graph-based Memory. Graph-based memory systems organize agent memory as a graph to capture structural dependencies among memory units. A-Mem(Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24)) constructs structured memory notes linked through LLM-assisted relation extraction and retrieves via seed selection followed by neighborhood expansion. Zep(Rasmussen et al., [2025](https://arxiv.org/html/2606.06036#bib.bib20)) maintains a bi-temporal knowledge graph that tracks when facts hold and invalidates outdated edges, supporting retrieval over evolving knowledge. LiCoMemory(Huang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib10)) organizes memory as a lightweight hierarchical graph that uses entities and relations as a semantic indexing layer, with temporal and hierarchy-aware search for efficient retrieval. While these representations improve relational and multi-hop access, traversal is governed by predefined operators and does not adapt to evidence accumulated during retrieval.

Hierarchical and Persistent Memory Systems. Hierarchical memory systems maintain long-lived memory that is continually updated across interactions. MemoryOS(Kang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib12)) organizes memory into a short, mid, and long-term hierarchy for structured access. Mem0(Chhikara et al., [2025](https://arxiv.org/html/2606.06036#bib.bib2)) maintains a compact set of salient facts through LLM-driven add, update, and delete operations. SeCom(Pan et al., [2025](https://arxiv.org/html/2606.06036#bib.bib18)) constructs the memory bank at the level of topically coherent segments for more accurate retrieval. These systems advance how memory is stored and updated over time, but their retrieval procedures remain passive, selecting memory units as a fixed function of the query without reasoning over intermediate evidence.

## 7 Conclusion and Discussion

We proposed MRAgent, a reconstructive memory agent that formulates memory access as an active, multi-step reconstruction process over a structured memory graph. A key design choice of MRAgent is to shift the modeling of complex relational dependencies to the retrieval stage, enabling the agent to resolve more complex queries with fewer computational cost through targeted, state-dependent exploration. As a result, our current implementation adopts a relatively simple memory construction strategy, without introducing additional complexity in memory updating or forgetting mechanisms. This design choice also exposes several limitations that suggest directions for future work. First, because relational reasoning is deferred to retrieval, the cost of reconstruction grows with the depth of exploration, and queries that require many traversal steps incur higher latency than single-shot retrieval. Second, our static construction does not update or consolidate memory over time, so the memory graph grows monotonically as interactions accumulate, raising storage overhead in long-lived deployments. Addressing these limitations through adaptive construction, lightweight memory maintenance, and more robust traversal policies is a promising avenue for extending active reconstruction to broader long-horizon settings.

## Acknowledgements

This research is supported by the Ministry of Education, Singapore, under the Academic Research Fund Tier 2 (FY2025) (Grant T2EP20124-0038).

## Impact Statement

This work aims to advance the design of memory systems for large language model (LLM) agents by introducing an active, reconstructive paradigm for long-horizon memory reasoning. Potential positive impacts include more reliable and context-aware AI assistants for applications such as personal assistants, decision support systems, and long-term human–AI interaction, where accurate recall and reasoning over past information are essential. At the same time, as with other memory-augmented AI systems, the ability to store and reason over long-term interaction data raises considerations around privacy, data governance, and responsible deployment. These concerns are not specific to our method but apply broadly to persistent memory in AI agents.

## References

*   Anokhin et al. (2024) Anokhin, P., Semenov, N., Sorokin, A., Evseev, D., Kravchenko, A., Burtsev, M., and Burnaev, E. Arigraph: Learning knowledge graph world models with episodic memory for llm agents. _arXiv preprint arXiv:2407.04363_, 2024. 
*   Chhikara et al. (2025) Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building production-ready AI agents with scalable long-term memory. _CoRR_, abs/2504.19413, 2025. doi: 10.48550/ARXIV.2504.19413. URL [https://doi.org/10.48550/arXiv.2504.19413](https://doi.org/10.48550/arXiv.2504.19413). 
*   Fang et al. (2025) Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y., Xu, Z., Deng, S., Yao, Y., Wang, M., Qiao, S., Chen, H., and Zhang, N. Lightmem: Lightweight and efficient memory-augmented generation. _CoRR_, abs/2510.18866, 2025. doi: 10.48550/ARXIV.2510.18866. URL [https://doi.org/10.48550/arXiv.2510.18866](https://doi.org/10.48550/arXiv.2510.18866). 
*   Frankland & Josselyn (2019) Frankland, P.W. and Josselyn, S.A. The neurobiological foundation of memory retrieval. _Nature Neuroscience_, 22(10):1576–1585, 2019. 
*   Gao et al. (2026) Gao, H., Geng, J., Hua, W., Hu, M., Juan, X., Liu, H., Liu, S., Qiu, J., Qi, X., Ren, Q., Wu, Y., Wang, H., Xiao, H., Zhou, Y., Zhang, S., Zhang, J., Xiang, J., Fang, Y., Zhao, Q., Liu, D., Qian, C., Wang, Z., Hu, M., Wang, H., Wu, Q., Ji, H., and Wang, M. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. _Trans. Mach. Learn. Res._, 2026, 2026. 
*   Han et al. (2025) Han, H., Wang, Y., Shomer, H., Guo, K., Ding, J., Lei, Y., Halappanavar, M., Rossi, R.A., Mukherjee, S., Tang, X., He, Q., Hua, Z., Long, B., Zhao, T., Shah, N., Javari, A., Xia, Y., and Tang, J. Retrieval-augmented generation with graphs (graphrag). _CoRR_, abs/2501.00309, 2025. doi: 10.48550/ARXIV.2501.00309. URL [https://doi.org/10.48550/arXiv.2501.00309](https://doi.org/10.48550/arXiv.2501.00309). 
*   Hatalis et al. (2023) Hatalis, K., Christou, D., Myers, J., Jones, S., Lambert, K., Amos-Binks, A., Dannenhauer, Z., and Dannenhauer, D. Memory matters: The need to improve long-term memory in llm-agents. In _Proceedings of the AAAI Symposium Series_, volume 2, pp. 277–280, 2023. 
*   Hendrycks et al. (2025) Hendrycks, D., Song, D., Szegedy, C., Lee, H., Gal, Y., Brynjolfsson, E., Li, S., Zou, A., Levine, L., Han, B., Fu, J., Liu, Z., Shin, J., Lee, K., Mazeika, M., Phan, L., Ingebretsen, G., Khoja, A., Xie, C., Salaudeen, O., Hein, M., Zhao, K., Pan, A., Duvenaud, D., Li, B., Omohundro, S., Alfour, G., Tegmark, M., McGrew, K., Marcus, G., Tallinn, J., Schmidt, E., and Bengio, Y. A definition of AGI. _CoRR_, abs/2510.18212, 2025. doi: 10.48550/ARXIV.2510.18212. URL [https://doi.org/10.48550/arXiv.2510.18212](https://doi.org/10.48550/arXiv.2510.18212). 
*   Hu et al. (2025) Hu, Y., Liu, S., Yue, Y., Zhang, G., Liu, B., Zhu, F., Lin, J., Guo, H., Dou, S., Xi, Z., et al. Memory in the age of ai agents. _arXiv preprint arXiv:2512.13564_, 2025. 
*   Huang et al. (2025) Huang, Z., Tian, Z., Guo, Q., Zhang, F., Zhou, Y., Jiang, D., and Zhou, X. Licomemory: Lightweight and cognitive agentic memory for efficient long-term reasoning. _CoRR_, abs/2511.01448, 2025. doi: 10.48550/ARXIV.2511.01448. URL [https://doi.org/10.48550/arXiv.2511.01448](https://doi.org/10.48550/arXiv.2511.01448). 
*   Jin et al. (2025) Jin, B., Zeng, H., Yue, Z., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. _CoRR_, abs/2503.09516, 2025. doi: 10.48550/ARXIV.2503.09516. URL [https://doi.org/10.48550/arXiv.2503.09516](https://doi.org/10.48550/arXiv.2503.09516). 
*   Kang et al. (2025) Kang, J., Ji, M., Zhao, Z., and Bai, T. Memory OS of AI agent. _CoRR_, abs/2506.06326, 2025. doi: 10.48550/ARXIV.2506.06326. URL [https://doi.org/10.48550/arXiv.2506.06326](https://doi.org/10.48550/arXiv.2506.06326). 
*   LangChain (2025) LangChain. Langmem sdk for agent long-term memory. [https://blog.langchain.com/langmem-sdk-launch/](https://blog.langchain.com/langmem-sdk-launch/), 2025. Accessed: 2025-01. 
*   Lewis et al. (2020) Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., and Kiela, D. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_, 2020. 
*   Li et al. (2025) Li, X., Dong, G., Jin, J., Zhang, Y., Zhou, Y., Zhu, Y., Zhang, P., and Dou, Z. Search-o1: Agentic search-enhanced large reasoning models. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025_, pp. 5420–5438. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.EMNLP-MAIN.276. URL [https://doi.org/10.18653/v1/2025.emnlp-main.276](https://doi.org/10.18653/v1/2025.emnlp-main.276). 
*   Maharana et al. (2024) Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. Evaluating very long-term conversational memory of LLM agents. In Ku, L., Martins, A., and Srikumar, V. (eds.), _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024_, pp. 13851–13870. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.747. URL [https://doi.org/10.18653/v1/2024.acl-long.747](https://doi.org/10.18653/v1/2024.acl-long.747). 
*   Manns et al. (2003) Manns, J.R., Hopkins, R.O., and Squire, L.R. Semantic memory and the human hippocampus. _Neuron_, 38(1):127–133, 2003. 
*   Pan et al. (2025) Pan, Z., Wu, Q., Jiang, H., Luo, X., Cheng, H., Li, D., Yang, Y., Lin, C.-Y., Zhao, H.V., Qiu, L., et al. Secom: On memory construction and retrieval for personalized conversational agents. In _The Thirteenth International Conference on Learning Representations_, 2025. 
*   Rashid et al. (2016) Rashid, A.J., Yan, C., Mercaldo, V., Hsiang, H.-L., Park, S., Cole, C.J., De Cristofaro, A., Yu, J., Ramakrishnan, C., Lee, S.Y., et al. Competition between engrams influences fear memory formation and recall. _Science_, 353(6297):383–387, 2016. 
*   Rasmussen et al. (2025) Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. Zep: A temporal knowledge graph architecture for agent memory. _CoRR_, abs/2501.13956, 2025. doi: 10.48550/ARXIV.2501.13956. URL [https://doi.org/10.48550/arXiv.2501.13956](https://doi.org/10.48550/arXiv.2501.13956). 
*   Rugg & Renoult (2025) Rugg, M.D. and Renoult, L. The cognitive neuroscience of memory representation. _Neuroscience & Biobehavioral Reviews_, 155:105450, 2025. 
*   Tan et al. (2025) Tan, Z., Yan, J., Hsu, I.-H., Han, R., Wang, Z., Le, L., Song, Y., Chen, Y., Palangi, H., Lee, G., et al. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pp. 8416–8439, 2025. 
*   Wu et al. (2025) Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K., and Yu, D. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In _The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025_. OpenReview.net, 2025. URL [https://openreview.net/forum?id=pZiyCaVuti](https://openreview.net/forum?id=pZiyCaVuti). 
*   Xu et al. (2025) Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-MEM: agentic memory for LLM agents. _CoRR_, abs/2502.12110, 2025. doi: 10.48550/ARXIV.2502.12110. URL [https://doi.org/10.48550/arXiv.2502.12110](https://doi.org/10.48550/arXiv.2502.12110). 
*   Zhong et al. (2024) Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In Wooldridge, M.J., Dy, J.G., and Natarajan, S. (eds.), _Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada_, pp. 19724–19731. AAAI Press, 2024. doi: 10.1609/AAAI.V38I17.29946. URL [https://doi.org/10.1609/aaai.v38i17.29946](https://doi.org/10.1609/aaai.v38i17.29946). 

## Appendix A Detailed Analysis of Passive Retrieval

This appendix provides a unified analysis of the retrieval mechanisms in representative memory systems for LLM agents, aligning all methods with the formulation in Section[2](https://arxiv.org/html/2606.06036#S2 "2 Problem Setting: Active Memory Access ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"). Recall that memory access operates over an external memory \mathcal{M} with units \mathcal{V}=\{v_{1},\ldots,v_{N}\}. Given a query x, passive retrieval returns a fixed set \pi_{\mathrm{p}}(x), while active reconstruction selects units sequentially via a stateful policy v^{(t)}=\pi_{\mathrm{a}}^{(t)}(x,S^{(t-1)}).

#### Retrieval-Augmented Generation.

Retrieval-augmented generation (RAG) adopts a classic similarity-based retrieval paradigm, where an external memory is organized as a vector store and queried in a single shot. Given a query x, RAG computes a relevance score between x and each memory unit v\in\mathcal{V}, and retrieves the top-k most similar items:

\pi_{\mathrm{p}}^{\textsc{RAG}}(x)=\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},\,k\right).(13)

A-Mem(Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24)). A-Mem introduces a graph structure over memory units, where nodes correspond to structured memory notes and edges encode semantic relations constructed during memory construction. Given a query x, A-Mem performs retrieval by first selecting a seed memory item via similarity, and then expanding its predefined neighborhood along the memory graph:

\displaystyle\mathcal{V}^{*}\displaystyle=\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},\,k\right),(14)
\displaystyle\pi_{\mathrm{p}}^{\textsc{A-Mem}}(x)\displaystyle=\mathcal{V}^{*}\ \cup\ \mathrm{Neighbor}(\mathcal{V}^{*}).

where \mathrm{Neighbor}(\mathcal{V}^{*})=\bigcup_{v\in\mathcal{V}^{*}}\mathrm{Neighbor}(v) returns the nodes directly connected to the seed set in the memory graph.

MemoryOS(Kang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib12)). MemoryOS organizes agent memory into a three-tier hierarchy with short-term memory (STM), mid-term memory (MTM), and long-term personal memories (LPM). Given a query x, it performs hierarchical retrieval by incorporating recent short-term context, relevant mid-term topic memories, and long-term persona information:

\displaystyle R_{\mathrm{MTM}}(x)\displaystyle=\mathrm{TopK}\!\left(\{\mathcal{F}_{score}(x,v)\}_{v\in\mathcal{V}_{\mathrm{MTM}}}\right),\quad R_{\mathrm{LPM}}^{\mathrm{fact}}(x)=\mathrm{TopK}\!\left(\{\mathcal{F}_{score}(x,v)\}_{v\in\mathcal{V}_{\mathrm{LPM}}^{\mathrm{fact}}}\right),(15)
\displaystyle\pi_{\mathrm{p}}^{\textsc{MemOS}}(x)\displaystyle=\mathcal{V}_{\mathrm{STM}}\;\cup\;R_{\mathrm{MTM}}(x)\;\cup\;\mathcal{V}_{\mathrm{LPM}}^{\mathrm{profile}}\;\cup\;R_{\mathrm{LPM}}^{\mathrm{fact}}(x),

where \mathcal{V}_{\mathrm{LPM}}^{\mathrm{profile}} represents persona information, and \mathcal{V}_{\mathrm{LPM}}^{\mathrm{fact}} denotes query-conditioned long-term factual memory.

LangMem(LangChain, [2025](https://arxiv.org/html/2606.06036#bib.bib13)). LangMem maintains conversational memory by compressing dialogue history into a set of memory summaries, which are stored in a vector-based memory store and injected into the model context during inference. Each memory unit corresponds to a summarized dialogue message represented by an embedding vector. Given a query x, LangMem retrieves relevant memories via similarity-based search:

\pi_{\mathrm{p}}^{\textsc{LangMem}}(x)=\mathcal{S}_{\textsc{LLM}}\!\left(\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},\,k\right),\,x\right).(16)

where \mathrm{Summarize}_{\textsc{LLM}}(\cdot) denotes an LLM-based summarization function that compresses retrieved memory entries into a concise context representation.

Mem0(Chhikara et al., [2025](https://arxiv.org/html/2606.06036#bib.bib2)). Mem0 maintains a persistent memory store composed of natural-language facts extracted incrementally from conversations via an LLM-driven memory construction and update process. Each memory item corresponds to a salient factual statement, and memory evolution is governed by LLM-selected operations (ADD, UPDATE, DELETE, NOOP) to ensure consistency over time. Given a query x, Mem0 performs retrieval by selecting the most relevant memory items via similarity search:

\pi_{\mathrm{p}}^{\textsc{Mem0}}(x)=\mathrm{TopK}\!\left(\{\mathrm{sim}(x,v)\}_{v\in\mathcal{V}},\,k\right).(17)

#### Conclusion.

From the above analysis, existing memory systems primarily focus on the design of memory representations, with retrieval procedures tightly coupled to the underlying structure. These approaches can be broadly categorized as similarity-based retrieval and graph-based retrieval. Although some methods employ structured memory representations, their retrieval processes remain passive: traversal operators are predefined and fixed, and do not adapt based on intermediate evidence during retrieval.

## Appendix B Implementation and Execution Details

### B.1 Memory Construction Pipeline

Cue–Tag–Episode. To instantiate elements of the memory graph and explicitly model their associative relations, we employ an LLM to extract the components required to construct Cue–Tag–Episode triplets. Given the raw dialogue text T, we first rewrite and refine it to resolve contextual dependencies and make cues explicit, and then segment the processed text into coherent episodic units:

\displaystyle\{e_{i}\}\leftarrow\mathcal{R}_{\text{LLM}}(T),(18)

where \mathcal{R}_{\text{LLM}} performs raw text processing, including pronoun resolution, temporal normalization, and episodic segmentation. For each episodic segment e_{i}, we generate an associative tag and a set of fine-grained cues using LLM-based extractors:

\displaystyle g_{i}\leftarrow\mathcal{T}_{\text{LLM}}(x_{i}),\quad\mathcal{C}_{i}\leftarrow\mathcal{K}_{\text{LLM}}(x_{i}),(19)

where \mathcal{T}_{\text{LLM}} produces a short and precise phrase summarizing the core semantic or relational pattern of the episode, and \mathcal{K}_{\text{LLM}} extracts a set of cues C_{i}, including entities, attributes, and other salient descriptors. We then add nodes for x_{i} and each c\in C_{i}, and associate them with the tag g_{i} to form the Cue–Tag–Episode relations.

Cue–Tag–Semantic. Semantic memory complements episodic reconstruction by providing relatively stable abstractions beyond individual episodes(Anokhin et al., [2024](https://arxiv.org/html/2606.06036#bib.bib1); Hu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib9)). Each semantic unit is represented as (c^{s}_{i},g^{s}_{i},s_{i}), where s_{i} denotes the semantic content, c^{s}_{i} is an entity-level cue, and g^{s}_{i} is an aspect-level tag. We extract such semantic units from the input text T via an LLM-based semantic extraction function:

\displaystyle\{(c^{s}_{i},g^{s}_{i},s_{i})\}\leftarrow\mathcal{S}_{\text{LLM}}(T),(20)

where \mathcal{S}_{\text{LLM}} identifies stable facts and attributes expressed in T, summarizes them into semantic content s_{i}, and assigns an entity-level cue c^{s}_{i} together with an aspect-level tag g^{s}_{i}. The extracted (c^{s}_{i},g^{s}_{i},s_{i}) units are inserted into the same memory graph and linked by Cue–Tag–Semantic associations.

High-level Topic Memory. While Tag captures episode-level semantic patterns, Topic summarizes recurrent patterns across episodes and provides a higher-level organization for multi-granular reasoning. This design is consistent with the view that higher-order structures can be abstracted from repeated episodic experiences(Pan et al., [2025](https://arxiv.org/html/2606.06036#bib.bib18); Tan et al., [2025](https://arxiv.org/html/2606.06036#bib.bib22)). Concretely, we obtain topic nodes by applying an LLM-based abstraction function to the set of episodic segments:

\displaystyle\{\tau_{j}\}\leftarrow\mathcal{A}_{\text{LLM}}(\{e_{i}\}),(21)

where each topic node \tau_{j} summarizes a coherent set of episodes that share recurring semantic patterns. We then add topic–episode links to connect each \tau_{j} with its associated episodes, enabling the agent to reason at a higher level when fine-grained episodic traversal is unnecessary.

Table 4: Memory toolkit providing operations for controlled memory traversal.

### B.2 Executable Memory Reconstruction

Figure[8](https://arxiv.org/html/2606.06036#A2.F8 "Figure 8 ‣ B.2 Executable Memory Reconstruction ‣ Appendix B Implementation and Execution Details ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") presents a detailed illustration of the memory reconstruction process, accompanied by an example diagram and pseudocode. To realize multi-step reasoning trajectories, we design a specialized toolkit that executes the traversal actions determined by the LLM and returns newly retrieved evidence from the memory graph.

MRAgent operates under two execution modes, \Psi\in\{\mathtt{Navigate},\mathtt{Answer}\}. In the \mathtt{Navigate} mode, the agent invokes the toolkit to progressively explore the memory graph and accumulate evidence. Once sufficient evidence has been collected, the agent transitions into the \mathtt{Answer} mode to generate the final response conditioned on the reconstructed context.

Table[4](https://arxiv.org/html/2606.06036#A2.T4 "Table 4 ‣ B.1 Memory Construction Pipeline ‣ Appendix B Implementation and Execution Details ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") summarizes the toolkit used for memory traversal and evidence acquisition. Each tool corresponds to a typed mapping between memory components, enabling the LLM to explicitly control the direction and granularity of memory access. Rather than retrieving a fixed set of memories, the agent composes these tools into sequential or parallel invocations to progressively reconstruct relevant context. At each navigation step, the LLM selects and invokes one or more tools based on the current reconstruction state. The returned results are treated as candidate evidence and are evaluated by the LLM to decide whether to expand, refine, or terminate a traversal path. This explicit tool-based interface ensures that memory access remains interpretable and controllable, while allowing the agent to adapt its retrieval strategy to intermediate evidence.

![Image 8: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/fig_example.png)

Algorithm 1 Active Memory Reconstruction

0: Query

x
, memory graph

\mathcal{G}
, traversal actions

\mathcal{A}=\{\Pi_{1},\dots,\Pi_{m}\}
, max steps

T

0: Answer

\hat{y}

1:

\mathcal{C}\leftarrow\textsc{ExtractCues}(x)

2:

\mathcal{Z}^{(0)}\leftarrow\textsc{ActiveSetInit}(\mathcal{C},\mathcal{G})

3:

\mathcal{H}^{(0)}\leftarrow\varnothing

4:

\mathcal{S}^{(0)}\leftarrow(\mathcal{Z}^{(0)},\mathcal{H}^{(0)})

5:for

t=0
to

T-1
do

6:// Action selection

7:

\mathcal{A}^{(t)}\leftarrow f_{\text{select}}(x,\mathcal{H}^{(t)},\mathcal{Z}^{(t)})
// \mathcal{A}^{(t)}\subseteq\mathcal{A}

8:// Controlled traversal

9:

\widetilde{\mathcal{Z}}^{(t+1)}\leftarrow\varnothing

10:for all

a\in\mathcal{A}^{(t)}
do

11:

\widetilde{\mathcal{Z}}^{(t+1)}\leftarrow\widetilde{\mathcal{Z}}^{(t+1)}\cup\Pi_{a}(\mathcal{Z}^{(t)})

12:end for

13:// Routing and state update

14:

\mathcal{Z}^{(t+1)}\leftarrow f_{\text{route}}(x,\mathcal{H}^{(t)},\widetilde{\mathcal{Z}}^{(t+1)})

15:

\mathcal{H}^{(t+1)}\leftarrow\mathcal{H}^{(t)}\cup\mathcal{Z}^{(t+1)}

16:if

\textsc{Stop}(x,\mathcal{H}^{(t+1)})
then

17:break

18:end if

19:end for

20:

\hat{y}\leftarrow\textsc{AnswerLLM}(x,\mathcal{H}^{(t+1)})

21:return

\hat{y}

Figure 8: Left: An illustrative example of memory reconstruction given a query. Right: Pseudocode of the memory reconstruction algorithm.

## Appendix C Theoretical Analysis: Active vs. Passive Retrieval over a Heterogeneous KG Memory

MRAgent is designed as an _active_ retrieval approach over a graph-based memory. This section formalizes an approximation-theoretic advantage of such _active_ (adaptive) retrieval over _passive_ (non-adaptive) retrieval.

### C.1 Setup

We consider tasks involving queries x\in\mathcal{X}, with answers y from a label space \mathcal{Y}.

Our MRAgent method uses a heterogeneous knowledge graph (KG) memory with cues, tags and episodes. However, for generality and notational simplicity, our theory generalizes this to arbitrary heterogeneous KGs with textual information on each node.

#### Heterogeneous KG memory.

For each problem size n, a memory M\in\mathcal{M}_{n} is a heterogeneous KG with node set

\mathcal{V}(M)=\{v_{1},v_{2},\dots,v_{n}\}.

Each node v\in\mathcal{V}(M) has a type \tau(v) in some type set, and a textual _payload_ p(v)\in\mathcal{P}. The KG also has a finite set of relation labels \mathcal{R} (e.g., tag types).

#### Node-retrieval operator.

Our LM interacts with the KG via a _retrieval operator_ that, given a node, returns both (i) the node payload and (ii) a bounded list of outgoing neighbors.

Formally, fix a branching bound B\geq 1. For each memory M, define

\mathrm{Retrieve}^{M}:\mathcal{V}(M)\to\mathcal{P}\times(\mathcal{R}\times\mathcal{V}(M))^{\leq B}.

Given a node v, the operator returns

\mathrm{Retrieve}^{M}(v)=\bigl(p(v),\ \mathrm{Out}(v)\bigr),

where p(v)\in\mathcal{P} is the payload and \mathrm{Out}(v) is a list of at most B outgoing edges (r,v^{\prime}), where r is the edge’s relation and v^{\prime} is its target node. This abstraction matches common KG implementations where a “node fetch” returns attributes/text plus a bounded set of neighbor references.

### C.2 Population risk and approximation error

###### Definition C.1(Population risk).

For a predictor \pi and distribution D, define the population risk under 0–1 loss:

L(\pi;D):=\mathbb{E}_{(M,x,y)\sim D}\bigl[\mathbf{1}[\pi(x,M)\neq y]\bigr].

###### Definition C.2(Approximation error).

For a hypothesis class \mathcal{H} and distribution D, define

\mathrm{opt}(\mathcal{H};D):=\inf_{\pi\in\mathcal{H}}L(\pi;D).

A policy that learns nothing about y can still guess using the prior. To capture this baseline under 0–1 loss, define

\varepsilon_{Y}\;:=\;1-\sup_{y\in\mathcal{Y}}P_{Y}(y).

Intuitively, \varepsilon_{Y} is the minimum error achievable when you only know the prior P_{Y} and have no additional information about y.

### C.3 Active vs. passive retrieval

We now define active and passive policies, given an LM with parameter \theta\in\Theta and a retrieval budget of T steps. The LM induces functions Q_{\theta,t} for t\leq T which decide which node to retrieve in step t, and a final answer head H_{\theta} which outputs an answer given the retrieval history.

Under the active policy, Q_{\theta,t} is conditioned on the entire history seen so far, but under the passive policy, it is conditioned only on the query. Let v^{(t)} denote the node the LM chooses to retrieve in step t. Formally:

###### Definition C.3(Active LM+retrieval policy class).

An _active_ policy with parameters \theta proceeds as:

v^{(1)}=Q_{\theta,1}(x),\qquad z^{(1)}=\mathrm{Retrieve}^{M}(v^{(1)}),

and for t=2,\dots,T,

v^{(t)}=Q_{\theta,t}(x,z^{(1)},\dots,z^{(t-1)}),\qquad z^{(t)}=\mathrm{Retrieve}^{M}(v^{(t)}).

The prediction is

\pi^{\mathrm{act}}_{\theta}(x,M)=H_{\theta}(x,z^{(1)},\dots,z^{(T)}).

Define the hypothesis class

\mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T):=\{\pi^{\mathrm{act}}_{\theta}:\theta\in\Theta\}.

###### Definition C.4(Passive LM+retrieval policy class).

A _passive_ policy with parameters \theta proceeds as:

v^{(t)}=Q^{\mathrm{pass}}_{\theta,t}(x),\qquad t=1,\dots,T,

then retrieves z^{(t)}=\mathrm{Retrieve}^{M}(v^{(t)}) for all t, and outputs

\pi^{\mathrm{pass}}_{\theta}(x,M)=H_{\theta}(x,z^{(1)},\dots,z^{(T)}).

Define

\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T):=\{\pi^{\mathrm{pass}}_{\theta}:\theta\in\Theta\}.

### C.4 Main theorem: active retrieval is strictly more powerful than passive

We show that allowing the retriever to adapt its queries to previously retrieved content yields a strictly larger class of predictors than any non-adaptive (query-only) retriever.

###### Theorem C.5(Active retrieval is strictly more powerful than passive).

For any retrieval budget T\geq 2, the passive hypothesis class is strictly contained in the active hypothesis class:

\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T)\ \subsetneq\ \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T).

To start, we first observe that passive policies are a subset of active policies, making the active hypothesis class _at least_ as strong as the passive class.

###### Lemma C.6.

For any retrieval budget T, we have

\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T)\ \subseteq\ \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T).

###### Proof.

Fix T and any passive policy \pi^{\mathrm{pass}}_{\theta}\in\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T). By definition, its retrieval queries take the form v^{(t)}=Q^{\mathrm{pass}}_{\theta,t}(x), depending only on the query x. Define an active policy \tilde{\pi}^{\mathrm{act}}_{\theta} with queries

\tilde{v}^{(t)}\;=\;\tilde{Q}_{\theta,t}(x,z^{(1)},\dots,z^{(t-1)})\;:=\;Q^{\mathrm{pass}}_{\theta,t}(x),\qquad t=1,\dots,T,

i.e., the active policy ignores the retrieval history and issues the same sequence of nodes as the passive one. Using the same answer head H_{\theta}, the resulting prediction function satisfies \tilde{\pi}^{\mathrm{act}}_{\theta}(x,M)=\pi^{\mathrm{pass}}_{\theta}(x,M) for all (x,M). ∎

#### What remains to be shown.

To prove strictness in Theorem 1 for any T\geq 2, it now suffices to construct a distribution D (depending on T) such that active retrieval can achieve zero error with budget T, while any passive policy with the same budget has error bounded away from zero, which we will show in the next subsection.

### C.5 A separating task family: Binary-Tree Needle-in-a-Haystack

We now construct a task with the required property. Intuitively, the query identifies a root node. Retrieval from the root reveals two candidate children; the root payload contains a single bit telling which child to follow. Repeating for d hops yields a unique target leaf, whose payload contains the answer label. Active retrieval can follow the revealed bits in order to achieve zero error, but passive retrieval must guess the leaf, incurring irreducible error.

#### Binary tree.

We form a complete binary tree of depth d=T-1. The root is denoted v_{\emptyset}. Each internal node v_{u} is indexed by a binary string u representing the path from the root to this node: starting from the root, each bit indicates whether to take the left (0) or right (1) child. As such, each internal node v_{u} has two children, labeled v_{u0} and v_{u1}.

#### Ground-truth leaf.

Sample a target leaf index u^{\star}\in\{0,1\}^{d} uniformly at random, and let the target node be the corresponding leaf v_{u^{\star}}. We interpret the bits of u^{\star}=(u^{\star}_{1},\dots,u^{\star}_{d}) as the ground-truth path from the root: starting at v_{\emptyset}, at depth t we follow the child indexed by u^{\star}_{t}\in\{0,1\}, so the unique root-to-leaf path is

v_{\emptyset}\;\to\;v_{u^{\star}_{1}}\;\to\;v_{u^{\star}_{1}u^{\star}_{2}}\;\to\;\cdots\;\to\;v_{u^{\star}_{1}\cdots u^{\star}_{d}}\;=\;v_{u^{\star}}.

To make this path discoverable by active retrieval, we encode the next bit along the target path in the payloads of nodes on the path: for each prefix u=u^{\star}_{1}\cdots u^{\star}_{t} with |u|=t<d,

p(v_{u})\ \text{encodes the next bit}\ u^{\star}_{t+1}.

#### Answer label.

Sample y\sim P_{Y}. The answer label is encoded _only_ in the target leaf payload:

p(v_{u^{\star}})\ \text{encodes}\ y.

All other returned information (payloads of non-target nodes and outgoing lists) is independent of y. In particular, unless a policy retrieves v_{u^{\star}}, it obtains no information about y beyond the prior.

#### Query.

The query x identifies the root node v_{\emptyset} and asks for the label y.

###### Definition C.7(Binary-Tree Needle-in-a-Haystack distribution D_{n,d}).

Fix d\geq 1 and set n as:

n\;=\;1+2+\cdots+2^{d}\;=\;2^{d+1}-1.

The distribution D_{n,d} over triples (M,x,y)\in\mathcal{M}_{n}\times\mathcal{X}_{n}\times\mathcal{Y} is defined as follows:

1.   1.
Sample a target leaf index u^{\star}\sim\mathrm{Unif}(\{0,1\}^{d}).

2.   2.
Sample y\sim P_{Y}.

3.   3.

Construct a heterogeneous KG memory M\in\mathcal{M}_{n} as the binary tree above:

    *   •
for each prefix u=u^{\star}_{1}\cdots u^{\star}_{t} with t<d, the payload p(v_{u}) encodes the next bit u^{\star}_{t+1};

    *   •
the payload p(v_{u^{\star}}) encodes y;

    *   •
all other node payloads are independent of y.

4.   4.
Output a query x that identifies v_{\emptyset} and asks for the label stored at v_{u^{\star}}.

### C.6 Active retrieval achieves zero error

###### Lemma C.8.

For the distribution D_{n,d}, there exists a parameter setting \theta and a budget T=d+1 such that

L\!\left(\pi^{\mathrm{act}}_{\theta};\,D_{n,d}\right)=0.

Consequently,

\mathrm{opt}\!\left(\mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(d+1);\,D_{n,d}\right)=0.

###### Proof.

Consider the following active strategy. The query x identifies the root node v_{\emptyset}. Initialize u:=\emptyset. For t=0,1,\dots,d-1:

1.   1.
Retrieve the current node v_{u} to obtain \mathrm{Retrieve}^{M}(v_{u})=(p(v_{u}),\mathrm{Out}(v_{u})). From p(v_{u}), read the next-bit value b\in\{0,1\} (which equals u^{\star}_{t+1} when u is the length-t prefix of u^{\star}).

2.   2.
Move to the corresponding child: set u\leftarrow ub, i.e., go to v_{u0} if b=0 and to v_{u1} if b=1.

After d steps we have reached the target leaf v_{u^{\star}}. Finally retrieve v_{u^{\star}} to read its payload, and output y. This uses at most d+1 retrievals and is correct for every sample (M,x,y)\sim D_{n,d}. ∎

### C.7 Passive retrieval has irreducible error unless the budget is exponential

###### Lemma C.9.

For any passive policy with budget T, uniformly over all parameters \theta,

L\!\left(\pi^{\mathrm{pass}}_{\theta};\,D_{n,d}\right)\ \geq\ \varepsilon_{Y}\!\left(1-\frac{T}{2^{d}}\right),\qquad\text{where}\qquad\varepsilon_{Y}:=1-\sup_{y\in\mathcal{Y}}P_{Y}(y).

###### Proof.

Fix \theta. A passive policy chooses all nodes v^{(1)},\dots,v^{(T)} to retrieve as functions of x alone, before observing any retrieved values.

Under D_{n,d}, the target leaf index u^{\star}\in\{0,1\}^{d} is uniform and is not revealed by x. Let S:=\{v^{(1)},\dots,v^{(T)}\} be the (multi)set of retrieved nodes, and let L_{d}:=\{v_{u}:\,u\in\{0,1\}^{d}\} be the set of leaves. Since only leaves can equal the target v_{u^{\star}},

\Pr\!\left[v_{u^{\star}}\in S\right]\;=\;\Pr\!\left[v_{u^{\star}}\in S\cap L_{d}\right]\;=\;\frac{|S\cap L_{d}|}{2^{d}}\;\leq\;\frac{T}{2^{d}}.

The label y is encoded only in the target payload p(v_{u^{\star}}). Therefore, the policy can obtain information about y only if it retrieves v_{u^{\star}}. Otherwise, the entire retrieval transcript is independent of y. In that case, the best possible prediction depends only on the prior P_{Y}, and must incur error at least \varepsilon_{Y}. Therefore,

L\!\left(\pi^{\mathrm{pass}}_{\theta};\,D_{n,d}\right)\ \geq\ \Pr[\text{miss target}]\cdot\varepsilon_{Y}\ \geq\ \left(1-\frac{T}{2^{d}}\right)\varepsilon_{Y}.

∎

### C.8 Strictness and approximation-theoretic separation

###### Proof of Theorem C.5.

Fix any T\geq 2 and set d:=T-1. The inclusion \mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T)\subseteq\mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T) follows by taking any passive policy and viewing it as an active policy whose query functions ignore the history and depend only on x (Lemma[C.6](https://arxiv.org/html/2606.06036#A3.Thmtheorem6 "Lemma C.6. ‣ C.4 Main theorem: active retrieval is strictly more powerful than passive ‣ Appendix C Theoretical Analysis: Active vs. Passive Retrieval over a Heterogeneous KG Memory ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents")).

For strictness, consider the separating distribution D_{n,d} from Definition C.7. By Lemma[C.8](https://arxiv.org/html/2606.06036#A3.Thmtheorem8 "Lemma C.8. ‣ C.6 Active retrieval achieves zero error ‣ Appendix C Theoretical Analysis: Active vs. Passive Retrieval over a Heterogeneous KG Memory ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), \mathrm{opt}(\mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T);D_{n,d})=0 (since T=d+1). By Lemma[C.9](https://arxiv.org/html/2606.06036#A3.Thmtheorem9 "Lemma C.9. ‣ C.7 Passive retrieval has irreducible error unless the budget is exponential ‣ Appendix C Theoretical Analysis: Active vs. Passive Retrieval over a Heterogeneous KG Memory ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"),

\mathrm{opt}(\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T);D_{n,d})\ \geq\ \varepsilon_{Y}\!\left(1-\frac{T}{2^{d}}\right)\ =\ \varepsilon_{Y}\!\left(1-\frac{T}{2^{T-1}}\right).

If the two hypothesis classes were equal, they would have equal \mathrm{opt}(\cdot;D) for every distribution D, contradicting the strict gap above. Hence the inclusion is strict for every T\geq 2. ∎

## Appendix D Experiments

### D.1 Datasets

LoCoMo(Maharana et al., [2024](https://arxiv.org/html/2606.06036#bib.bib16)). LoCoMo is a benchmark designed to evaluate long-term conversational memory in extended dialogue settings. The dataset consists of 50 conversations generated via a human–LLM pipeline. Each conversation spans up to 35 sessions, with an average length of approximately 300 turns, and is accompanied by roughly 200 question–answer pairs. LoCoMo includes multiple question categories, covering single-hop, multi-hop, temporal, and open-domain queries, which require retrieving and integrating information from distant parts of the conversation history. In our experiments, we exclude adversarial questions, as most baseline methods do not support this setting and the task primarily evaluates the ability to detect unanswerable queries rather than memory reconstruction or multi-hop reasoning.

LongMemEval(Wu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib23)). LongMemEval is a benchmark designed to evaluate very long-term memory capabilities of chat assistants under sustained user–assistant interactions. Each evaluation instance consists of a sequence of timestamped chat sessions, followed by a question that requires recalling and reasoning over the interaction history. We adopt the LongMemEval-S setting, which contains approximately 500 questions, each paired with a chat history of around 115K tokens. This setting provides a challenging evaluation of long-term memory. In our experiments, we primarily focus on the single-session-user multi-session, temporal-reasoning and single-session-preference question types.

### D.2 Baselines

Retrieval-Augmented Generation. Retrieval-augmented generation(RAG) stores dialogue histories as text chunks in a vector database. For each query, the system retrieves the top-k most similar memory entries based on embedding similarity and concatenates them with the query to form the input context.

A-Mem(Xu et al., [2025](https://arxiv.org/html/2606.06036#bib.bib24)). A-Mem constructs structured memory notes for each conversational episode, where new memory notes are linked to existing ones using LLM-assisted analysis, resulting in a directed graph structure. For retrieval, A-MEM embeds the query to retrieve relevant memory notes and then expands to their neighboring nodes.

MemoryOS(Kang et al., [2025](https://arxiv.org/html/2606.06036#bib.bib12)). MemoryOS organizes agent memory into a three-tier hierarchy, including short-term memory for recent dialogue context, mid-term memory for topic-based interaction history, and long-term persona memory for stable user and agent characteristics. For retrieval, it performs hierarchical memory access across all layers and the retrieved memories are concatenated with persona information to form the input context.

LangMem(LangChain, [2025](https://arxiv.org/html/2606.06036#bib.bib13)). LangMem stores conversational memories by embedding each dialogue turn and maintaining them in a vector database via a memory management tool. At inference time, it retrieves the top-k most relevant memories based on embedding similarity and summarizes them using the LLM to support response generation.

Mem0(Chhikara et al., [2025](https://arxiv.org/html/2606.06036#bib.bib2)). Mem0 incrementally extracts salient natural-language facts from conversations and maintains a compact long-term memory through LLM-driven update operations. For retrieval, it performs similarity-based search over stored memories and incorporates the retrieved facts into the input context for answer generation.

### D.3 Evaluation Metric

LLM-Judge. We adopt an LLM-based judge to evaluate the semantic correctness of generated answers. Given a model-generated answer \hat{y} and a reference answer y, the judge determines whether \hat{y} is semantically equivalent to y, allowing for paraphrasing and differences in surface form. The judge outputs a binary decision indicating correctness.

F 1 Score. In addition to Recall, we report the F 1 score computed from the judge decisions. For each question, the generated answer is treated as a positive prediction, and the judge decision determines whether it is a true positive or a false positive. Aggregating over the dataset, precision and recall are computed as:

\mathrm{Precision}=\frac{\sum_{i}J(\hat{y}_{i},y_{i})}{\sum_{i}\mathbb{I}[\hat{y}_{i}\neq\emptyset]},\qquad\mathrm{Recall}=\frac{\sum_{i}J(\hat{y}_{i},y_{i})}{N},(22)

and the F 1 score is defined as:

\mathrm{F}_{1}=\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.(23)

This formulation penalizes both incorrect answers and failures to produce an answer, and is particularly suitable for open-ended question answering settings.

Evidence Recall (Recall). Evidence Recall measures the fraction of ground-truth supporting evidence that is successfully retrieved during memory reconstruction. For each question i, let \mathcal{E}_{i}^{\ast} denote the set of annotated ground-truth evidence items, and let \hat{\mathcal{E}}_{i} denote the set of evidence items retrieved by the agent. Evidence Recall is computed as:

\mathrm{Recall}=\frac{1}{N}\sum_{i=1}^{N}\frac{|\hat{\mathcal{E}}_{i}\cap\mathcal{E}_{i}^{\ast}|}{|\mathcal{E}_{i}^{\ast}|},(24)

where N is the total number of evaluation questions. This metric reflects the effectiveness of the retrieval process in recovering relevant supporting evidence, independent of final answer generation.

### D.4 Implementation

We use GPT-4o-mini as the LLM judge with temperature set to 0.0. Each method is evaluated three independent times, and we report the mean and standard deviation of the judge scores across runs. To ensure comparable compute budgets across methods, we cap the agent’s reasoning to at most 8 turns per query, and allow up to 10 tool invocations per turn. Agents may terminate early if a stopping condition is met before exhausting the budget.

### D.5 Detailed Results on LongMemEval

Table[5](https://arxiv.org/html/2606.06036#A4.T5 "Table 5 ‣ D.5 Detailed Results on LongMemEval ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") presents detailed results on LongMemEval under different evaluation settings, evaluated by F 1 and LLM-Judge.

Table 5: Performance on LongMemEval evaluated by F 1 and LLM-Judge\uparrow. Best and second-best results among Gemini-backbone methods are marked in bold and underline, respectively. MRAgent∗ uses Claude for retrieval while memories are constructed by Gemini.

### D.6 Budget Sensitivity Analysis of Multi-step Reconstruction

![Image 9: Refer to caption](https://arxiv.org/html/2606.06036v1/Fig/depth_call_heatmap.png)

Figure 9:  Performance on multi-hop queries in LoCoMo as a function of the number of reasoning turns (T) and the per-round retrieval budget (K), evaluated under the Claude backbone using LLM-Judge (J). 

Figure[9](https://arxiv.org/html/2606.06036#A4.F9 "Figure 9 ‣ D.6 Budget Sensitivity Analysis of Multi-step Reconstruction ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") analyzes the trade-off between the number of reasoning turns and per-turn parallel retrieval on multi-hop questions in LoCoMo. We vary the per-turn retrieval budget (K), corresponding to the maximum number of parallel tool calls allowed within a single reasoning turn, and the maximum number of reasoning turns (T), while keeping other settings fixed.

Reconstruction depth cannot be substituted by increased parallel exploration. As the number of reasoning turns T increases, accuracy improves steadily and monotonically across all values of K, with deeper reconstruction yielding substantially higher performance. In contrast, increasing the per-turn retrieval budget K leads to only limited gains that quickly saturate. These results indicate that while parallel exploration increases retrieval breadth within a single reasoning turn, it cannot replace the sequential composition of evidence enabled by multi-turn reconstruction.

### D.7 Evidence Coverage by Retrieval Operators

Table 6: Evidence coverage of different tools across question types on LoCoMo under the Claude backbone.

To analyze how different retrieval operators contribute to memory reconstruction, we examine the evidence coverage of each tool on LoCoMo, aggregated by question category. Table[6](https://arxiv.org/html/2606.06036#A4.T6 "Table 6 ‣ D.7 Evidence Coverage by Retrieval Operators ‣ Appendix D Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents") reports the coverage rates of individual operators, reflecting their functional roles during reconstruction.

Different retrieval operators specialize in distinct query structures. Temporal questions are predominantly resolved through query_conversation_time, which accounts for the majority of temporally grounded evidence. Multi-hop questions rely heavily on query_tag_events and query_topic_events, indicating that associative expansion over tags and topics is essential for recovering evidence distributed across multiple episodes. In contrast, open-domain questions exhibit more balanced coverage across multiple operators, reflecting the need to integrate episodic, semantic, and contextual information. Overall, these results demonstrate that MRAgent performs structured, operator-dependent memory reconstruction. Rather than relying on uniform or similarity-driven retrieval, the agent selectively activates different operators based on query structure, resulting in differentiated evidence acquisition patterns.

### D.8 Case Studies

As shown in Figure[7](https://arxiv.org/html/2606.06036#S5.F7 "Figure 7 ‣ 5.6 Case Study (RQ5) ‣ 5 Experiments ‣ Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"), we present a case study illustrating how MRAgent performs multi-turn memory reconstruction over a structured memory graph. The query, “Which of Joanna’s screenplays were rejected by production companies?”, requires linking screenplay submission events with subsequent rejection events that are distributed across multiple dialogue sessions.

MRAgent incrementally reconstructs relevant evidence through iterative graph exploration and multi-turn reasoning. In the first reasoning turn, the agent retrieves screenplay submission and rejection events by traversing tag-based associations, identifying candidate events that are directly related to the query. In the second turn, it expands these candidates by querying event-level context and keywords to obtain detailed information about each rejection. In the third and fourth turns, the agent queries semantic information about Joanna to recover properties of her screenplays, enabling a clearer characterization of each submission. Finally, in the fifth turn, the agent queries temporal information to align screenplay submissions with corresponding rejection events and verify their ordering.

After five reasoning steps, MRAgent correctly infers that Joanna’s first and third screenplays were rejected. This example demonstrates that answering complex, multi-session queries benefits from active, multi-step memory reconstruction that integrates associative expansion with semantic and temporal verification.

## Appendix E Prompt

