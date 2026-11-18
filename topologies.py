"""Multi-agent communication topology executors supporting Plain-Python and LangGraph backends."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from agents import AgentOutput, BaseAgent
from config import TopologyBackend, TopologyType


def run_chain_plain(agents: List[BaseAgent], initial_input: str) -> List[AgentOutput]:
    """Execute sequential pipeline where each agent processes the output of the preceding agent."""
    outputs: List[AgentOutput] = []
    current_input = initial_input

    for agent in agents:
        agent_out = agent.execute(current_input)
        outputs.append(agent_out)
        current_input = agent_out.output_text

    return outputs


def run_star_plain(agents: List[BaseAgent], initial_input: str) -> List[AgentOutput]:
    """Execute star fan-out where satellite agents independently process the central hub's output."""
    if len(agents) < 2:
        raise ValueError("Star topology requires at least 2 agents (Hub + >=1 satellite).")

    outputs: List[AgentOutput] = []

    hub_agent = agents[0]
    hub_output = hub_agent.execute(initial_input)
    outputs.append(hub_output)

    for satellite in agents[1:]:
        sat_output = satellite.execute(hub_output.output_text)
        outputs.append(sat_output)

    return outputs


def run_ring_plain(agents: List[BaseAgent], initial_input: str, max_hops: Optional[int] = None) -> List[AgentOutput]:
    """Execute cyclical feedback ring where agents pass outputs in a round-robin sequence."""
    outputs: List[AgentOutput] = []
    current_input = initial_input
    num_agents = len(agents)
    total_steps = max_hops if max_hops is not None else (num_agents + 1)

    for step in range(total_steps):
        agent = agents[step % num_agents]
        agent_out = agent.execute(current_input, hop_index=step)
        if step >= num_agents:
            agent_out.agent_name = f"{agent.name}_loop"
        outputs.append(agent_out)
        current_input = agent_out.output_text

    return outputs


def run_tree_plain(agents: List[BaseAgent], initial_input: str) -> List[AgentOutput]:
    """Execute hierarchical tree topology with root, branch, and leaf node levels."""
    if len(agents) < 5:
        raise ValueError("Tree topology requires at least 5 agents (Root + 2 Branches + 2 Leaves).")

    root = agents[0]
    branch_1 = agents[1]
    branch_2 = agents[2]
    leaf_1 = agents[3]
    leaf_2 = agents[4]

    out_root = root.execute(initial_input, hop_index=0)
    out_b1 = branch_1.execute(out_root.output_text, hop_index=1)
    out_b2 = branch_2.execute(out_root.output_text, hop_index=1)
    out_l1 = leaf_1.execute(out_b1.output_text, hop_index=2)
    out_l2 = leaf_2.execute(out_b2.output_text, hop_index=2)

    return [out_root, out_b1, out_b2, out_l1, out_l2]


def merge_outputs_dict(
    existing: Dict[str, AgentOutput],
    incoming: Dict[str, AgentOutput]
) -> Dict[str, AgentOutput]:
    """Merge dictionary state updates from parallel LangGraph nodes without key collisions."""
    merged = dict(existing)
    merged.update(incoming)
    return merged


class GraphState(TypedDict):
    """Shared state container for LangGraph multi-agent graph workflows."""
    task_input: str
    outputs: Annotated[Dict[str, AgentOutput], merge_outputs_dict]


def build_langgraph_chain(agents: List[BaseAgent]) -> Any:
    """Compile a linear LangGraph StateGraph connecting agents sequentially."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as e:
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    workflow = StateGraph(GraphState)

    for i, agent in enumerate(agents):
        def make_node(ag: BaseAgent, prev_name: Optional[str]):
            def node_func(state: GraphState) -> Dict[str, Any]:
                if prev_name is None:
                    step_input = state["task_input"]
                else:
                    step_input = state["outputs"][prev_name].output_text
                result = ag.execute(step_input)
                return {"outputs": {ag.name: result}}
            return node_func

        prev_agent_name = agents[i - 1].name if i > 0 else None
        workflow.add_node(agent.name, make_node(agent, prev_agent_name))

    workflow.add_edge(START, agents[0].name)
    for i in range(len(agents) - 1):
        workflow.add_edge(agents[i].name, agents[i + 1].name)
    workflow.add_edge(agents[-1].name, END)

    return workflow.compile()


def build_langgraph_star(agents: List[BaseAgent]) -> Any:
    """Compile a LangGraph StateGraph with fan-out edges from a hub agent to satellites."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as e:
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    workflow = StateGraph(GraphState)
    hub_agent = agents[0]
    satellite_agents = agents[1:]

    def hub_node(state: GraphState) -> Dict[str, Any]:
        result = hub_agent.execute(state["task_input"])
        return {"outputs": {hub_agent.name: result}}

    workflow.add_node(hub_agent.name, hub_node)

    for sat in satellite_agents:
        def make_sat_node(ag: BaseAgent):
            def sat_func(state: GraphState) -> Dict[str, Any]:
                hub_out = state["outputs"][hub_agent.name].output_text
                result = ag.execute(hub_out)
                return {"outputs": {ag.name: result}}
            return sat_func

        workflow.add_node(sat.name, make_sat_node(sat))

    workflow.add_edge(START, hub_agent.name)
    for sat in satellite_agents:
        workflow.add_edge(hub_agent.name, sat.name)
        workflow.add_edge(sat.name, END)

    return workflow.compile()


def build_langgraph_ring(agents: List[BaseAgent]) -> Any:
    """Compile a LangGraph StateGraph representing a feedback ring with loop closure."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as e:
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    workflow = StateGraph(GraphState)
    num_agents = len(agents)

    ag0 = agents[0]
    def node0(state: GraphState) -> Dict[str, Any]:
        res = ag0.execute(state["task_input"], hop_index=0)
        return {"outputs": {ag0.name: res}}

    workflow.add_node(ag0.name, node0)
    workflow.add_edge(START, ag0.name)

    prev_name = ag0.name
    for i in range(1, num_agents):
        ag = agents[i]
        def make_node(curr_ag: BaseAgent, p_name: str, hop: int):
            def node_func(state: GraphState) -> Dict[str, Any]:
                res = curr_ag.execute(state["outputs"][p_name].output_text, hop_index=hop)
                return {"outputs": {curr_ag.name: res}}
            return node_func

        workflow.add_node(ag.name, make_node(ag, prev_name, i))
        workflow.add_edge(prev_name, ag.name)
        prev_name = ag.name

    loop_name = f"{ag0.name}_loop"
    def loop_node(state: GraphState) -> Dict[str, Any]:
        res = ag0.execute(state["outputs"][prev_name].output_text, hop_index=num_agents)
        res.agent_name = loop_name
        return {"outputs": {loop_name: res}}

    workflow.add_node(loop_name, loop_node)
    workflow.add_edge(prev_name, loop_name)
    workflow.add_edge(loop_name, END)

    return workflow.compile()


def build_langgraph_tree(agents: List[BaseAgent]) -> Any:
    """Compile a LangGraph StateGraph representing a hierarchical tree with branch and leaf levels."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as e:
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    if len(agents) < 5:
        raise ValueError("Tree topology requires at least 5 agents (Root + 2 Branches + 2 Leaves).")

    workflow = StateGraph(GraphState)
    root, b1, b2, l1, l2 = agents[0], agents[1], agents[2], agents[3], agents[4]

    def root_node(state: GraphState) -> Dict[str, Any]:
        result = root.execute(state["task_input"], hop_index=0)
        return {"outputs": {root.name: result}}

    def b1_node(state: GraphState) -> Dict[str, Any]:
        res = b1.execute(state["outputs"][root.name].output_text, hop_index=1)
        return {"outputs": {b1.name: res}}

    def b2_node(state: GraphState) -> Dict[str, Any]:
        res = b2.execute(state["outputs"][root.name].output_text, hop_index=1)
        return {"outputs": {b2.name: res}}

    def l1_node(state: GraphState) -> Dict[str, Any]:
        res = l1.execute(state["outputs"][b1.name].output_text, hop_index=2)
        return {"outputs": {l1.name: res}}

    def l2_node(state: GraphState) -> Dict[str, Any]:
        res = l2.execute(state["outputs"][b2.name].output_text, hop_index=2)
        return {"outputs": {l2.name: res}}

    workflow.add_node(root.name, root_node)
    workflow.add_node(b1.name, b1_node)
    workflow.add_node(b2.name, b2_node)
    workflow.add_node(l1.name, l1_node)
    workflow.add_node(l2.name, l2_node)

    workflow.add_edge(START, root.name)
    workflow.add_edge(root.name, b1.name)
    workflow.add_edge(root.name, b2.name)
    workflow.add_edge(b1.name, l1.name)
    workflow.add_edge(b2.name, l2.name)
    workflow.add_edge(l1.name, END)
    workflow.add_edge(l2.name, END)

    return workflow.compile()


def run_langgraph_topology(
    topology_type: TopologyType,
    agents: List[BaseAgent],
    initial_input: str
) -> List[AgentOutput]:
    """Execute a compiled LangGraph topology graph and extract ordered agent outputs."""
    if topology_type == TopologyType.CHAIN:
        graph = build_langgraph_chain(agents)
    elif topology_type == TopologyType.STAR:
        graph = build_langgraph_star(agents)
    elif topology_type == TopologyType.RING:
        graph = build_langgraph_ring(agents)
    elif topology_type == TopologyType.TREE:
        graph = build_langgraph_tree(agents)
    else:
        raise ValueError(f"Unsupported topology type for LangGraph: {topology_type}")

    initial_state: GraphState = {
        "task_input": initial_input,
        "outputs": {}
    }

    final_state = graph.invoke(initial_state)
    outputs_map = final_state["outputs"]

    if topology_type == TopologyType.RING:
        ordered_names = [ag.name for ag in agents] + [f"{agents[0].name}_loop"]
        return [outputs_map[name] for name in ordered_names if name in outputs_map]
    elif topology_type == TopologyType.TREE:
        ordered_names = [agents[0].name, agents[1].name, agents[2].name, agents[3].name, agents[4].name]
        return [outputs_map[name] for name in ordered_names if name in outputs_map]

    return [outputs_map[ag.name] for ag in agents if ag.name in outputs_map]


def execute_topology(
    topology: TopologyType,
    agents: List[BaseAgent],
    task_input: str,
    backend: TopologyBackend = TopologyBackend.PLAIN_PYTHON
) -> List[AgentOutput]:
    """Dispatch execution across topologies and backends, returning ordered agent outputs."""
    if backend == TopologyBackend.LANGGRAPH:
        return run_langgraph_topology(topology, agents, task_input)

    if topology == TopologyType.CHAIN:
        return run_chain_plain(agents, task_input)
    elif topology == TopologyType.STAR:
        return run_star_plain(agents, task_input)
    elif topology == TopologyType.RING:
        return run_ring_plain(agents, task_input)
    elif topology == TopologyType.TREE:
        return run_tree_plain(agents, task_input)
    else:
        raise ValueError(f"Unsupported topology type: {topology}")
