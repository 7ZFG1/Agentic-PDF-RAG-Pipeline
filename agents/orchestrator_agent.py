import os
import sys

from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VERTEXAI_PROJECT, VERTEXAI_LOCATION, GEMINI_MODEL

from agents.retriever_agent import RetrieverAgent
from agents.image_analyst_agent import ImageAnalystAgent
from agents.validator_agent import ValidatorAgent

def _create_tools(retriever_agent: RetrieverAgent, image_analyst_agent: ImageAnalystAgent) -> list:
    """Create LangChain tools that wrap our agents."""

    @tool
    def search_chunks(query: str) -> str:
        """Search the document for relevant text and image chunks based on a query."""
        return retriever_agent(query)

    @tool
    def analyze_image(image_path: str, question: str) -> str:
        """Analyze a document image to extract visual information."""
        return image_analyst_agent(image_path, question)

    return [search_chunks, analyze_image]


class OrchestratorAgent:
    """Coordinates the ReAct loop using LangGraph."""

    def __init__(self, retriever_agent: RetrieverAgent, image_analyst_agent: ImageAnalystAgent, validator_agent: ValidatorAgent) -> None:
        self.validator_agent = validator_agent
        self.max_iterations = 10

        # create tools from agents
        self.tools = _create_tools(retriever_agent, image_analyst_agent)

        # LangChain LLM with tools bound
        self.llm = ChatVertexAI(
            model_name=GEMINI_MODEL,
            project=VERTEXAI_PROJECT,
            location=VERTEXAI_LOCATION
        ).bind_tools(self.tools)

        # build LangGraph
        self.graph = self._build_graph()

    def __call__(self, question: str) -> str:
        """Run full orchestration loop and return validated answer."""
        print(f"  [Orchestrator] Starting pipeline for: '{question}'")

        system_msg = SystemMessage(content=(
            "You are a document QA assistant. Answer the user's question using ONLY information "
            "obtained from the available tools (search_chunks and analyze_image). Do not use "
            "outside knowledge.\n\n"
            "Tools:\n"
            "- search_chunks(query): searches the document's text and image index and returns a "
            "JSON list of chunks with relevance scores ('score'). Higher score = more relevant.\n"
            "- analyze_image(image_path, question): sends one image to a vision model with a "
            "focused question and returns its analysis.\n\n"
            "Follow these steps:\n"
            "1. Call search_chunks with a query based on the user's question.\n"
            "2. Evaluate the results:\n"
            "   - If you got relevant, high-scoring text chunks that answer the question, go to step 4.\n"
            "   - If the results are empty, low-scoring, or off-topic, call search_chunks AGAIN with "
            "a reformulated query (use synonyms, more specific or broader terms, key entity/model "
            "names, or translate key terms between Turkish and English, since the document may be "
            "in either language). Retry like this up to 2 times in total.\n"
            "   - If after retries you still have nothing useful, proceed to step 4 with what you have.\n"
            "3. For image chunks returned by search_chunks:\n"
            "   - First check the chunk's 'caption' and 'description' fields — these often already "
            "answer simple questions about the image.\n"
            "   - Call analyze_image ONLY if the image is relevant to the question AND the text "
            "chunks plus existing caption/description are not sufficient (e.g., the question asks "
            "about specific numbers, values, or details inside a chart/table/diagram).\n"
            "   - Ask analyze_image a focused question about exactly what you need from that image.\n"
            "   - Do not call analyze_image more than once for the same image_path.\n"
            "4. Produce the final answer:\n"
            "   - Base it strictly on the retrieved chunks and image analyses.\n"
            "   - If, after your search attempts, the document does not contain information to "
            "answer the question, say so explicitly instead of guessing.\n"
            "   - Answer in the same language the user asked the question in."
        ))
        human_msg = HumanMessage(content=question)

        # run the graph with recursion limit
        limit = self.max_iterations
        print(f"  [Orchestrator] Running graph (recursion_limit={limit})...")

        try:
            result = self.graph.invoke(
                {"messages": [system_msg, human_msg]},
                config={"recursion_limit": limit}
            )
        except Exception as e:
            print(f"  [Orchestrator] Graph error: {e}")
            return f"Error during orchestration: {e}"

        # extract draft answer (last AI message)
        draft = result["messages"][-1].content
        print(f"  [Orchestrator] Draft answer received ({len(draft)} chars)")

        # collect context from tool responses for validation
        context = self._extract_context(result["messages"])
        print(f"  [Orchestrator] Collected {len(context)} chars of context for validation")

        # always validate
        final = self.validator_agent(question, draft, context)
        print(f"  [Orchestrator] Final answer ready")
        return final

    def _agent_node(self, state: MessagesState) -> dict:
        """LangGraph node: call LLM with current messages."""
        print(f"  [Orchestrator] Agent thinking... ({len(state['messages'])} messages)")
        
        response = self.llm.invoke(state["messages"])

        has_tools = hasattr(response, "tool_calls") and response.tool_calls
        print(f"  [Orchestrator] Agent decided: {'tool_call' if has_tools else 'final_answer'}")
        return {"messages": [response]}

    def _should_continue(self, state: MessagesState) -> str:
        """LangGraph edge: route to tools or end."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            print(f"  [Orchestrator] Routing -> tools")
            return "tools"
        print(f"  [Orchestrator] Routing -> end")
        return "end"

    def _extract_context(self, messages: list[BaseMessage]) -> str:
        """Pull tool response content from message history."""
        parts = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                parts.append(str(msg.content))
        return "\n---\n".join(parts)

    def _build_graph(self) -> object:
        """Build the LangGraph ReAct graph."""
        print(f"  [Orchestrator] Building LangGraph...")

        graph = StateGraph(MessagesState)

        # nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        # edges
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._should_continue, {
            "tools": "tools",
            "end": END
        })
        graph.add_edge("tools", "agent")

        compiled = graph.compile()
        print(f"  [Orchestrator] Graph compiled")
        return compiled
