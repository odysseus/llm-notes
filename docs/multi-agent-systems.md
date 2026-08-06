---
title: "Multi-Agent Systems"
type: concept
status: active
updated: 2026-08-05
tags: [agents, multi-agent-systems, orchestration, workflows, delegation, evaluation]
---

# Multi-Agent Systems: When Multiple LLM Agents Help

**Central idea:** A multi-agent system divides a task among two or more independently configured model-driven processes. This is useful when the division creates real parallelism, specialization, context isolation, independent evidence, or separate authority. It is not automatically better than a single agent, and additional agents can make sequential work slower, more expensive, and less reliable.

**Why it matters:** Multi-agent architectures are easy to make impressive in a diagram and difficult to justify in a product. Choosing well requires identifying what each additional agent contributes that could not be obtained more simply through a tool, a deterministic workflow, another model call, a larger context, or a stronger single-agent harness.

## Background topics

- **Agents and harnesses:** The distinction between a model, a goal-directed agent, and the runtime that controls it.
- **Tool calling:** How model proposals become validated operations in an external environment.
- **Workflow orchestration:** Sequential, branching, parallel, and event-driven control flow.
- **Task decomposition:** Dividing a larger objective into subtasks with clear inputs and outputs.
- **Concurrency and distributed systems:** Shared state, synchronization, retries, race conditions, and partial failure.
- **Context engineering:** Selecting the information and instructions available to each model call.
- **Ensembles and test-time computation:** Generating more than one candidate and selecting or combining the results.
- **Verification and evaluation:** Checking outputs against tests, evidence, schemas, or external state.
- **Security boundaries:** Assigning tools, data access, and permissions according to role.

These topics explain the mechanisms beneath multi-agent systems. The central design question is not how to connect agents, but whether separating the work improves the system after coordination costs are counted.

## Before LLM-based multi-agent systems

Multi-agent systems long predate large language models. Classical distributed artificial intelligence studied collections of software entities that perceive an environment, pursue objectives, communicate, negotiate, or coordinate. Some systems were cooperative, while others represented actors with partially conflicting interests. The Contract Net Protocol, for example, described a manager announcing a task and other nodes bidding to perform it—a recognizable ancestor of manager–worker agent designs ([Smith, 1980](https://doi.org/10.1109/TC.1980.1675516)).

Traditional agents usually operated within a carefully specified world. Their messages had defined meanings, available actions were known in advance, and planning or negotiation algorithms were designed for a particular problem. A warehouse robot, auction bidder, or game-playing agent did not need to interpret an arbitrary request written in natural language. The advantage was precision; the cost was that every new domain required substantial modeling and software.

LLMs changed the economics of building flexible agents. A language model could interpret underspecified instructions, write plans, call tools through textual schemas, summarize observations, and communicate with another model in ordinary language. The same underlying model could be prompted as a researcher, planner, critic, programmer, or coordinator. Researchers quickly began arranging multiple model instances into role-playing societies and organizational workflows.

CAMEL explored autonomous cooperation between role-playing chat agents ([Li et al., 2023](https://arxiv.org/abs/2303.17760)). AutoGen generalized multi-agent conversations into a programmable framework in which agents could combine models, humans, tools, and code-defined interaction patterns ([Wu et al., 2023](https://arxiv.org/abs/2308.08155)). MetaGPT encoded software-development roles and standard operating procedures into a structured multi-agent process ([Hong et al., 2023](https://arxiv.org/abs/2308.00352)). ChatDev similarly organized communicative agents into a software-company metaphor ([Qian et al., 2023](https://arxiv.org/abs/2307.07924)).

These systems established that an LLM application did not have to be one continuous conversation with one model persona. Work could be partitioned among separately prompted model calls, and the outputs of one role could become the inputs of another.

The early examples also encouraged an overly simple analogy: if human organizations benefit from managers, designers, programmers, reviewers, and testers, perhaps an LLM system should reproduce those job titles. The analogy is incomplete. Human roles often represent years of different training, private knowledge, social accountability, and independently formed judgment. Five agents using the same model, the same evidence, and slightly different role prompts may be five samples from one underlying system rather than five genuine experts.

Later research began measuring this distinction. *Why Do Multi-Agent LLM Systems Fail?* examined five popular systems across more than 150 tasks and organized observed problems into failures of specification and system design, inter-agent misalignment, and task verification or termination ([Cemri et al., 2025](https://arxiv.org/abs/2503.13657)). *Towards a Science of Scaling Agent Systems* compared single-agent and multi-agent topologies under standardized tools and token budgets. It found strong dependence on task structure: centralized coordination helped parallelizable work, while multi-agent approaches degraded tightly sequential reasoning tasks and incurred substantial overhead on tool-heavy tasks ([Kim et al., 2025](https://arxiv.org/abs/2512.08296)).

The mature question is therefore not “How many agents should this application have?” It is:

> What useful difference is created by giving this part of the work a separate context, policy, capability, state, or attempt?

## The topic in one view

An LLM agent can be described abstractly as a model embedded in a controlled loop:

$$
A_i = (M_i, C_i, T_i, S_i, P_i)
$$

where:

- $M_i$ is the model used by agent $i$;
- $C_i$ is its instructions and current context;
- $T_i$ is its available tool set;
- $S_i$ is its private or shared state;
- $P_i$ is the policy that determines how it acts, communicates, and stops.

A multi-agent system contains several such processes plus an orchestration mechanism:

$$
Y = O(A_1, A_2, \ldots, A_n, E)
$$

Here, $E$ is the external environment and $O$ controls task assignment, messages, shared artifacts, execution order, conflict resolution, and final aggregation.

The formula highlights an important point: adding agents also enlarges the harness. The system now needs communication protocols, handoff formats, budgets, shared-state rules, termination conditions, and a method for deciding which agent to trust. Multi-agent design is therefore a form of [harness engineering](harness-engineering.md), not an alternative to it.

The main sources of potential value are:

1. **Parallelism:** Independent subtasks can run at the same time.
2. **Specialization:** Agents can receive different models, tools, instructions, data, or permissions.
3. **Isolation:** Each agent can work within a smaller, cleaner context.
4. **Diversity:** Independent attempts can explore different hypotheses or solution paths.
5. **Verification:** One process can test or challenge another using independent evidence.
6. **Representation:** Separate agents can stand for genuinely distinct actors, users, or objectives.

The main costs are:

1. **Communication overhead:** Information must be compressed, transmitted, and interpreted.
2. **Coordination errors:** Work may be duplicated, omitted, ordered incorrectly, or assigned poorly.
3. **Error propagation:** One agent's unsupported claim can become another agent's premise.
4. **Compute cost:** More agents usually mean more tokens, model calls, and tool operations.
5. **Latency:** Sequential conversations and review rounds extend the critical path.
6. **Debugging complexity:** Failures emerge from interactions rather than one trace.

A multi-agent system is justified when its benefits are tied to observable properties of the task and exceed these costs.

## What counts as multiple agents?

The term *agent* is used loosely. A system does not become meaningfully multi-agent merely because it makes several LLM calls.

Consider four designs:

1. One model extracts fields, then the same application asks it to write a response.
2. One agent creates three candidate solutions and chooses among them.
3. Three independent workers investigate different sources and submit findings to a coordinator.
4. Personal agents representing different users negotiate without revealing their private constraints.

The first is a multi-step workflow. The second is an ensemble or search procedure. The third is clearly a delegated multi-agent system. The fourth is multi-agent in a stronger sense because the agents represent distinct information and authority boundaries.

There is no universal line between these categories, and a single software framework may express all of them using “agents.” Operationally, the multi-agent label is most useful when the components possess some meaningful independence:

- separate objectives or assigned responsibilities;
- separately constructed contexts;
- different tools, models, data, or permissions;
- local state that is not automatically shared;
- the ability to choose or execute more than one step;
- an explicit communication or handoff protocol.

If two named roles see the same full transcript, use the same model and tools, and merely take turns completing a fixed prompt sequence, the architecture is closer to a workflow with role prompts. That may still be useful, but its value should be attributed to decomposition and additional inference rather than to an imagined society of distinct minds.

## When multiple agents provide real value

### 1. Independent work can run in parallel

Parallelism is the clearest case. Suppose a research task requires separate investigations of technical feasibility, market evidence, regulation, and security. If each inquiry can proceed largely independently, four workers can search concurrently and return structured reports to a coordinator.

The benefit is not only potential answer quality. If external searches dominate runtime, parallel execution can reduce elapsed time even while total computation increases. The task must have a useful **parallel fraction**, however. If every step requires the exact output of the preceding step, more workers have nothing independent to do.

Parallel delegation works best when:

- subtask boundaries are evident;
- each worker receives sufficient local context;
- outputs have a common schema;
- duplicate coverage is either intentional or inexpensive;
- a final integrator can reconcile the results;
- workers do not concurrently mutate the same fragile state.

Research, broad test generation, independent data extraction, and evaluating many candidate items are often amenable to this pattern. A single linear proof, tightly coupled debugging session, or sequence of dependent tool calls usually is not.

### 2. Specialists genuinely differ

Specialization is valuable when it changes capabilities, not just labels. One agent might use a model optimized for visual understanding, another a code model, and another a less expensive model for classification. An agent may have access to a particular database, sandbox, or domain-specific retrieval index. A compliance reviewer may have read-only access and a policy corpus that the execution agent does not receive.

This is **heterogeneous specialization**:

| Difference | Example benefit |
| --- | --- |
| Model | Match coding, vision, long-context, or low-cost classification to the subtask |
| Tools | Give a researcher search tools and an executor transaction tools |
| Data | Isolate private customer context from public research context |
| Permissions | Separate proposal from approval or execution authority |
| Prompt and examples | Supply domain-specific procedures without crowding every context |
| State | Let agents maintain local progress without polluting a global transcript |

A role prompt alone can produce some behavioral specialization, but it does not give the model knowledge absent from its context or eliminate errors shared by the base model. The strongest specialist roles are grounded in different resources or verification methods.

### 3. Contexts need to remain isolated

Large tasks can overwhelm a single context with raw observations, conflicting instructions, and irrelevant details. Separate agents can each receive a bounded view and return a concise artifact. This reduces context interference and lets the coordinator operate on results rather than every intermediate token.

Isolation can also be a privacy or safety feature. A payroll agent may calculate from private compensation data and return only an aggregate to a planning agent. A user-representative agent may evaluate options using private preferences without exposing those preferences to the group.

Context isolation creates an information bottleneck, so handoff quality becomes critical. The worker should report evidence, confidence, assumptions, unresolved questions, and exact artifact references—not merely a persuasive summary. If the coordinator needs a fact that was omitted, the system needs a way to request it without broadcasting the worker's entire context.

### 4. Independent attempts improve coverage

One model response represents one sampled trajectory through a large possibility space. Independent attempts can explore alternative decompositions, hypotheses, or designs. Their value comes from diversity: the probability that at least one attempt discovers a good path may rise with the number and variety of attempts.

This is closely related to **self-consistency**, which samples several reasoning paths and selects the most consistent answer ([Wang et al., 2022](https://arxiv.org/abs/2203.11171)). It is also related to **Tree of Thoughts**, which explicitly searches among intermediate possibilities and backtracks ([Yao et al., 2023](https://arxiv.org/abs/2305.10601)). Neither method requires persistent role-playing agents. If all that is needed is a set of independent candidates, an ensemble or search algorithm is often simpler than a conversational team.

Independence must be protected. If later agents see the first answer before forming their own, they can anchor on it. Different prompts, models, evidence partitions, or random samples may increase diversity. The outputs should ideally be compared by an objective test or a separately designed evaluator.

### 5. Another agent can challenge or verify a proposal

Multi-agent debate asks models to propose answers, inspect one another's reasoning, and revise over several rounds. Du and colleagues reported improvements in mathematical reasoning and factuality from this procedure ([Du et al., 2023](https://arxiv.org/abs/2305.14325)). A related layered design, Mixture-of-Agents, gives aggregators the outputs of several models and reported strong results on language-model evaluation benchmarks ([Wang et al., 2024](https://arxiv.org/abs/2406.04692)).

The useful mechanism is not conversation itself. It is the introduction of additional candidates, criticism, and selection. Verification is strongest when the reviewer can obtain evidence that the generator did not use:

- run tests against generated code;
- recompute a numerical result;
- retrieve the cited source and check the claim;
- validate output against a schema or business invariant;
- compare a proposed change with authoritative application state.

A second LLM asked only “Does this look correct?” may repeat the first model's error or accept confident nonsense. A verifier with tests, evidence, or deliberately independent reasoning provides a more meaningful signal.

### 6. The problem contains genuinely distinct actors

Some tasks are multi-agent by nature rather than by implementation preference. Negotiation, games, markets, social simulations, group scheduling, and privacy-preserving coordination contain multiple participants with different information or objectives.

Generative Agents demonstrated a simulation in which language-model-driven characters maintained memories, formed plans, and interacted within a shared environment ([Park et al., 2023](https://arxiv.org/abs/2304.03442)). Such systems are not merely trying to produce one better answer. They study or implement interaction among actors.

The same principle applies to an application coordinating several real users. A personal agent can represent one user's private availability and preferences; a group coordinator can receive only permitted conclusions. Collapsing all users into one context would erase the desired privacy and authority boundaries. Here, separate agents model a real structure in the problem.

## Common multi-agent architectures

### Manager–worker hierarchy

A manager decomposes the objective, assigns subtasks, monitors progress, and integrates worker outputs.

This structure is useful when decomposition is dynamic but centralized oversight is desirable. The manager can limit duplication and preserve a global view. It also becomes a bottleneck: a poor decomposition contaminates every downstream task, and too much information may accumulate in the manager's context.

**Primary gain:** controlled parallel delegation.

**Primary risk:** central planning and aggregation failure.

### Sequential specialist pipeline

Work passes through ordered roles such as analyst → designer → implementer → tester. MetaGPT and ChatDev popularized versions of this pattern for software development.

Pipelines make ownership and handoffs explicit. They are appropriate when intermediate artifacts have stable definitions. If the sequence is fully known, however, it may be better described and implemented as a deterministic workflow whose stages happen to use different prompts or models. Giving every stage an open-ended agent loop may add autonomy where none is needed.

**Primary gain:** modular context and stage-specific instructions.

**Primary risk:** information loss and error accumulation across handoffs.

### Independent candidates with an aggregator

Several agents solve the same problem without seeing one another's outputs. A voting rule, verifier, or aggregator selects or synthesizes the result.

This pattern creates diversity without requiring long conversations. It works well when candidate quality can be ranked and when additional inference is worth the cost. Majority vote is most useful when errors are not strongly correlated and there is a well-defined answer. Synthesis is harder because an aggregator may combine incompatible fragments into a result that no worker actually supported.

**Primary gain:** broader search and robustness to one bad attempt.

**Primary risk:** correlated errors or an unreliable aggregator.

### Debate or critic–revision

Agents see and challenge one another's proposals before a final judge or consensus step. Debate can surface assumptions and counterarguments that a one-pass answer misses. It is attractive for ambiguous judgments, adversarial review, and tasks where objections are informative.

Debate can also produce conformity, endless discussion, or persuasion without truth. Shared model biases remain shared, and a correct minority can be talked out of its answer. Limit rounds, preserve original independent answers, require evidence, and define how disagreement is resolved.

**Primary gain:** explicit error discovery and comparison of reasoning.

**Primary risk:** social-style dynamics that reward confidence or agreement rather than correctness.

### Shared workspace or blackboard

Agents communicate mainly by reading and writing structured artifacts: a task board, database, document set, code repository, or event log. A coordinator may assign work, or agents may claim available tasks.

This often scales better than broadcasting every message to every agent. The shared environment becomes the source of truth, while conversations remain local. It requires normal distributed-systems discipline: ownership, versioning, locks or conflict detection, idempotent operations, and clear completion states.

**Primary gain:** durable coordination through artifacts rather than transcripts.

**Primary risk:** stale reads, conflicting writes, and hidden shared-state assumptions.

### Decentralized peer network

Agents communicate directly and decide locally what to do next. This can adapt to changing environments and avoid a single coordinator. It can also produce duplicate work, inconsistent beliefs, and difficult-to-reconstruct trajectories.

Decentralization is justified when no agent can maintain a complete global view, when the topology is part of the problem, or when resilience to a failed coordinator matters. It is usually excessive for a small business workflow.

**Primary gain:** local adaptation and removal of a central bottleneck.

**Primary risk:** coordination overhead and weak global control.

## Alternatives to a multi-agent architecture

Many proposed multi-agent use cases are better served by a simpler form of additional computation.

| Need | Simpler approach | When it is preferable |
| --- | --- | --- |
| Perform a known sequence | Deterministic workflow | Stages and transitions are predictable |
| Use several capabilities | One agent with several tools | One context can coordinate the work coherently |
| Improve a draft | Generator followed by validator or editor | Only one bounded review pass is needed |
| Explore alternatives | Independent samples or self-consistency | Candidates need no persistent roles or conversation |
| Search a branching solution space | Tree/beam search | States and candidate scores can be represented explicitly |
| Match requests to expertise or cost | Model/tool router | Only one specialist needs to run for each request |
| Save model cost | Model cascade | Easy cases can be accepted before escalating |
| Supply missing knowledge | Retrieval or a domain tool | The problem is information access, not task coordination |
| Handle a long task | One resumable agent with structured state | Work is sequential and benefits from continuity |
| Enforce correctness | Tests, schemas, rules, or a solver | Success can be checked deterministically |
| Improve raw capability | A stronger model or better context | Failures come from insufficient reasoning or missing information |

### One agent with tools

A single agent can already search, calculate, query databases, write files, and revise its plan. It retains one coherent view of the objective and avoids inter-agent handoffs. For most conversational assistants and CRUD-style applications, this should be the baseline.

If the agent's context becomes cluttered, improve retrieval, summarize tool results, or expose fewer tools per turn before introducing more agents. The problem may be poor context engineering rather than insufficient organizational structure.

### Deterministic workflows

A workflow can use LLMs for ambiguous steps while keeping the overall sequence in code:

```text
classify request
    → retrieve authoritative record
    → extract proposed changes
    → validate fields
    → request confirmation if required
    → write transaction
    → verify stored result
```

This is usually safer than asking a manager agent to invent and delegate the same sequence. Workflows provide exact transitions, straightforward tests, and predictable cost. Use agents where the next step genuinely depends on open-ended discoveries.

### Additional calls without additional agents

A generator–critic–revision sequence can improve a high-value response while remaining one bounded workflow. Self-consistency can produce multiple attempts without inventing biographies or persistent roles. Tree search can explore alternatives while keeping state and evaluation explicit.

These approaches capture much of the value commonly attributed to multi-agent systems: more test-time computation, diverse candidates, and error checking. They avoid much of the open-ended coordination burden.

### Routers and cascades

If requests require different expertise, a router can select one model, tool set, or prompt configuration. It is unnecessary to activate every specialist and ask them to discuss each request. RouterBench formalized evaluation of multi-LLM routing systems across model quality and cost tradeoffs ([Hu et al., 2024](https://arxiv.org/abs/2403.12031)).

A cascade begins with a cheaper model and escalates only when its result is uncertain or fails validation. This provides heterogeneous capability without simultaneous collaboration. The difficulty moves into routing and confidence estimation, but the runtime can be far lower than consulting a full panel for every input.

## A practical decision framework

Start with the strongest reasonable single-agent or workflow baseline. Then evaluate the task along the following dimensions.

| Dimension | Low value favors | High value favors |
| --- | --- | --- |
| Parallelizability | One agent or sequential workflow | Parallel workers |
| Subtask independence | Shared continuous context | Delegated contexts |
| Capability heterogeneity | One general model | Specialist models, tools, or data |
| Need for independent judgment | One generated answer | Ensemble, debate, or verifier |
| Objective verifiability | Direct generation may suffice | Candidate generation plus tests or judge |
| Shared-state contention | Centralized execution | Carefully partitioned ownership |
| Privacy or authority separation | One context and permission set | Separate representative agents |
| Communication burden | Multi-agent may be manageable | One coherent agent is often safer |
| Sequential dependency | Parallel workers idle or miscoordinate | Single agent preserves continuity |
| Cost and latency sensitivity | Minimal calls and routing | Extra agents only for high-value cases |

A compact decision sequence is:

1. **Can ordinary software perform the step?** Use code, a query, a solver, or a deterministic tool.
2. **Is the control flow known?** Use a workflow with model-powered nodes.
3. **Does one agent have enough context and capability?** Keep one agent unless evaluation shows a specific failure.
4. **Can work proceed independently?** Use parallel workers with structured outputs.
5. **Do roles need different tools, data, models, or authority?** Use genuine specialists.
6. **Is the goal more attempts rather than more roles?** Use sampling, search, or an ensemble.
7. **Can another process verify with independent evidence?** Add a bounded verifier.
8. **Does the environment contain multiple actors?** Represent them separately when their information or objectives must remain distinct.

The burden of proof should be attached to each additional agent. “Reviewer” is not a justification. “Read-only agent with access to the policy corpus that checks each proposed transaction and returns cited violations” is.

## Designing a multi-agent system

### Give every agent a contract

An agent contract should state:

- its objective and non-objectives;
- the inputs it may assume;
- the tools and data it may access;
- the artifact or schema it must return;
- whether it may mutate external state;
- its call, token, time, and retry budget;
- the evidence required to claim completion;
- how it reports uncertainty or blockage.

Role names can help humans read the design, but contracts make it executable. “Act as a senior analyst” is vague. “Return up to five claims, each with a source URL, quoted evidence location, confidence, and unresolved contradiction” is testable.

### Prefer artifacts to conversational relays

Long chains of agents paraphrasing one another behave like a game of telephone. Important qualifications disappear while confident wording survives. Pass structured artifacts whenever possible:

```json
{
  "subtask": "privacy terms",
  "status": "complete",
  "claims": [
    {
      "claim": "Provider does not train on API data by default",
      "source": "https://example.com/policy",
      "evidence_location": "Data use, paragraph 2",
      "confidence": 0.92
    }
  ],
  "conflicts": [],
  "open_questions": ["Enterprise retention period varies by plan"]
}
```

The coordinator should be able to trace a final statement back to a worker's evidence. Shared documents, database rows, code commits, and test results are stronger handoffs than prose buried in a group chat.

### Centralize consequential side effects

Parallel read operations are usually easier than parallel writes. Let workers research, propose changes, or prepare patches, while a controlled integration step owns external mutations. If workers must write concurrently, partition resources clearly and use version checks, transactions, or isolated workspaces.

The model assigned to a role should not automatically receive the authority suggested by its title. A “manager” can delegate tasks without credentials to send messages or spend money. A “reviewer” should usually be unable to silently rewrite the item it is reviewing.

### Make communication selective

Full broadcast is simple but grows expensive: every message becomes context for every participant. Selective routing sends only relevant artifacts to the agents that need them. A coordinator or typed event bus can enforce this.

Communication design should answer:

- who can contact whom;
- whether messages are requests, facts, proposals, or commands;
- which state is authoritative;
- how versions and dependencies are identified;
- what happens when agents disagree or do not respond;
- how loops are detected and stopped.

### Use explicit aggregation

The final answer should not emerge accidentally from whichever agent speaks last. Define whether outputs are concatenated, voted on, ranked, verified, reconciled, or synthesized. Each rule implies different assumptions.

Majority voting assumes that independent agents are more likely than not to be correct and that their errors are not too correlated. A judge assumes that candidate comparison is easier and more reliable than generation. Synthesis assumes compatible outputs whose provenance can be retained. Tests avoid these assumptions when correctness is mechanically checkable.

## A compact implementation pattern

The following framework-neutral pseudocode illustrates parallel research with centralized synthesis:

```python
async def answer_research_question(question):
    plan = planner.decompose(
        question,
        schema=List[ResearchAssignment],
        max_assignments=4,
    )

    assignments = validate_independent_scopes(plan)

    reports = await run_in_parallel([
        research_worker.run(
            objective=item.objective,
            allowed_sources=item.allowed_sources,
            output_schema=EvidenceReport,
            budget=Budget(calls=6, seconds=90),
        )
        for item in assignments
    ])

    accepted = []
    for report in reports:
        checked = verify_citations_and_schema(report)
        if checked.valid:
            accepted.append(checked.report)

    draft = synthesizer.run(
        question=question,
        evidence=accepted,
        requirement="Every factual claim must retain provenance",
    )

    return final_validator.check(draft)
```

The important properties are outside the model prompts:

- decomposition is bounded;
- workers run only after their scopes are validated;
- parallel work is read-only;
- outputs follow a shared evidence schema;
- citation checks occur before synthesis;
- the synthesizer sees accepted artifacts rather than every worker transcript;
- a final validator applies explicit completion criteria.

The planner, workers, and synthesizer could use the same model or different models. Their separate agent identities matter less than the task boundaries and evidence flow.

## Applying the choice to a task-and-idea chatbot

For the single-user MVP described in this research project, a multi-agent architecture is probably the wrong starting point. Capturing, retrieving, and updating an event are short, stateful operations with authoritative database records. One conversational agent inside a typed harness can interpret the request and use narrow tools such as `create_event`, `find_events`, and `update_event`.

Dividing “add oil change to my list” among a planner, database expert, critic, and coordinator would add calls without creating parallel work or genuine specialization. A deterministic validation and confirmation path is more reliable than an agent debate over the fields.

Multi-agent patterns may become useful in later, distinct parts of the larger vision:

- **Standing research:** parallel workers could investigate different event sources or categories, then a coordinator could deduplicate and rank suggestions.
- **Complex plan generation:** independent workers could develop travel, cost, and logistics options when those inquiries require different tools.
- **Group planning:** personal agents could represent separate users and reveal only approved preferences or constraints to a group coordinator.
- **Adversarial review:** a privacy-policy agent with separate policy data could check proposed information sharing before it occurs.

The group-planning case is the strongest. Separate agents would correspond to real users, private context, and permission boundaries. The research case is conditional: ordinary parallel functions or retrieval queries may be sufficient unless each worker needs its own adaptive tool loop.

## Common failure modes

### Decorative specialization

Agents receive job titles but no different information, tools, models, or verification methods. The system pays for several versions of similar reasoning and mistakes stylistic diversity for expertise.

### Premature decomposition

The coordinator splits the task before understanding it. Workers efficiently answer the wrong subquestions, and the final integrator cannot recover missing dependencies.

### Fragmented context

No agent has enough information to make a locally correct decision. Each produces a plausible partial result, but their assumptions conflict when combined.

### Telephone-game handoffs

Agents pass prose summaries rather than source-linked artifacts. Details and uncertainty disappear at every transition, while errors become increasingly polished.

### Correlated consensus

Several agents agree because they share the same model, prompt framing, and evidence. Agreement is treated as independent confirmation even though the failures are correlated.

### Error amplification

One agent invents a fact or marks a task complete; later agents accept it as an observation. The claim gains authority merely by traveling through the system. Controlled studies of agent scaling have found that architecture affects how strongly errors propagate, with centralized checking containing failures better than unchecked independent coordination in the tested settings.

### Unreliable judging

An aggregator prefers fluent, detailed, or agreeable answers over correct ones. The system spends heavily on candidate generation and then discards the best candidate.

### Endless refinement

Critics always find something to revise, agents repeatedly delegate the same task, or disagreement never reaches a termination rule. Every conversation needs budgets and externally defined stopping conditions.

### Shared-state conflict

Agents edit the same document, database record, or repository without ownership rules. One worker overwrites another, reads stale state, or verifies an outdated version.

### Authority leakage

Every role receives every tool for convenience. A research or review agent can then cause side effects unrelated to its responsibility. Tool permissions should follow the contract, not the breadth of the framework.

### Evaluation by unequal compute

A five-agent system is compared with one cheap one-pass baseline and credited with an architectural improvement that may come simply from using five or twenty times more inference. Equal-budget comparisons and strong single-agent baselines are necessary.

## Evaluating whether multiple agents helped

Evaluate the full system against the simplest credible alternative, not against an intentionally weak prompt.

At minimum, compare:

1. one model call;
2. one tool-using agent with a competent harness;
3. a deterministic or branching workflow;
4. a same-budget ensemble or search method;
5. the proposed multi-agent architecture.

Hold tools, available evidence, and total token or monetary budgets constant where the comparison requires it. Also report unconstrained operational performance when latency or parallel throughput is the intended benefit. Equal-cost accuracy and real-world elapsed time answer different questions.

Useful metrics include:

| Dimension | Example measure |
| --- | --- |
| Outcome quality | Task success, factual accuracy, tests passed, user acceptance |
| Parallel benefit | Wall-clock time relative to total compute |
| Coordination efficiency | Useful subtask output per message or token |
| Coverage | Relevant issues or hypotheses discovered |
| Redundancy | Duplicate work and repeated tool calls |
| Error propagation | Incorrect claims reused by downstream agents |
| Handoff quality | Required fields, provenance, and assumptions preserved |
| Reliability | Success rate across repeated trials |
| Cost | Tokens, model calls, tools, and infrastructure |
| Safety | Unauthorized actions, data exposure, or approval bypasses |
| Recoverability | Ability to retry or resume one failed component |

Ablations identify the real source of improvement. Remove the debate rounds but retain multiple samples. Replace specialist agents with the same model and prompt. Give a single agent the combined tools. Replace the LLM judge with tests. If performance remains unchanged, the multi-agent story may not explain the gain.

The evaluation should also include adversarial coordination cases:

- one worker returns a confident false claim;
- two workers produce contradictory results;
- a manager omits a required subtask;
- a tool times out after an ambiguous write;
- a worker exceeds its budget;
- shared state changes between research and execution;
- an agent attempts to access another role's private data.

Multi-agent systems should be judged as distributed applications whose components are probabilistic, not as a collection of independent chat demos.

## Practical design principles

- **Begin with one agent or a workflow.** Add agents in response to measured limitations.
- **Name the source of value.** Parallelism, specialization, isolation, diversity, verification, and representation are distinct benefits.
- **Prefer real heterogeneity.** Different tools, evidence, permissions, or models create stronger specialization than personas alone.
- **Keep dependent work together.** Tight sequential reasoning usually benefits from one coherent context.
- **Parallelize reads before writes.** Centralize or partition consequential mutations.
- **Use typed handoffs.** Preserve evidence, versions, assumptions, confidence, and unresolved questions.
- **Protect independence.** Do not expose candidates to one another before independent reasoning when diversity matters.
- **Verify outside the conversation.** Tests and authoritative state are stronger than agreement.
- **Make aggregation explicit.** Voting, judging, synthesis, and deterministic checking are not interchangeable.
- **Limit communication.** Send each agent only the context and artifacts it needs.
- **Budget the whole system.** More agents multiply tokens, retries, and failure opportunities.
- **Evaluate against equal-compute alternatives.** Extra inference is not automatically an architectural gain.
- **Treat coordination as product logic.** Version, trace, test, and inspect it like any other critical code.

## What multi-agent systems do not solve

Multiple agents do not create missing knowledge unless at least one can access better information. They do not turn subjective agreement into factual verification. They do not remove hallucinations; they can repeat, propagate, or amplify them. They do not automatically provide security, memory, or reliable tool use—those remain responsibilities of the harness.

They also do not guarantee diversity. Identical models often share biases and failure modes. Different personas can change style more than substance. Independent sources, models, tools, or sampling processes are more defensible reasons to expect complementary outputs.

Finally, multi-agent systems do not remove the need for a clear owner of the final result. Even decentralized work must eventually affect a user, file, database, or environment. The application needs a policy for authority, conflict, verification, and stopping.

## Recap

A multi-agent system is a collection of separately configured model-driven processes coordinated toward a task. It can outperform a single agent when the task contains independent parallel work, genuinely heterogeneous specialists, context or permission boundaries, useful candidate diversity, independent verification, or multiple real actors.

The same architecture can underperform when work is tightly sequential, tools dominate the budget, handoffs discard information, agents share correlated errors, or no reliable aggregator exists. Extra agents create additional inference and a distributed-systems problem at the same time.

The best default is therefore not maximum collaboration. It is the smallest architecture that exposes the required capability:

- use ordinary code for deterministic operations;
- use workflows for known sequences;
- use one agent for coherent adaptive work;
- use sampling or search for additional solution paths;
- use a verifier when independent checking is available;
- use multiple agents when separation itself creates measurable value.

The decisive question is not whether several agents can talk. It is whether their separation changes what the system can know, do, verify, protect, or accomplish in parallel.

## Key sources

- Smith (1980), [*The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver*](https://doi.org/10.1109/TC.1980.1675516).
- Dafoe et al. (2021), [*Cooperative AI: Machines Must Learn to Find Common Ground*](https://www.nature.com/articles/d41586-021-01170-0).
- Wang et al. (2022), [*Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171).
- Li et al. (2023), [*CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society*](https://arxiv.org/abs/2303.17760).
- Park et al. (2023), [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442).
- Yao et al. (2023), [*Tree of Thoughts: Deliberate Problem Solving with Large Language Models*](https://arxiv.org/abs/2305.10601).
- Du et al. (2023), [*Improving Factuality and Reasoning in Language Models through Multiagent Debate*](https://arxiv.org/abs/2305.14325).
- Qian et al. (2023), [*ChatDev: Communicative Agents for Software Development*](https://arxiv.org/abs/2307.07924).
- Hong et al. (2023), [*MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework*](https://arxiv.org/abs/2308.00352).
- Wu et al. (2023), [*AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*](https://arxiv.org/abs/2308.08155).
- Guo et al. (2024), [*Large Language Model Based Multi-Agents: A Survey of Progress and Challenges*](https://arxiv.org/abs/2402.01680).
- Hu et al. (2024), [*RouterBench: A Benchmark for Multi-LLM Routing System*](https://arxiv.org/abs/2403.12031).
- Wang et al. (2024), [*Mixture-of-Agents Enhances Large Language Model Capabilities*](https://arxiv.org/abs/2406.04692).
- Kapoor et al. (2024), [*AI Agents That Matter*](https://arxiv.org/abs/2407.01502).
- Anthropic (2024), [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents).
- Cemri et al. (2025), [*Why Do Multi-Agent LLM Systems Fail?*](https://arxiv.org/abs/2503.13657).
- Kim et al. (2025), [*Towards a Science of Scaling Agent Systems*](https://arxiv.org/abs/2512.08296).
