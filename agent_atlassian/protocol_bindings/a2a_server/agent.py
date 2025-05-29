# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import asyncio
from typing_extensions import override
from typing import Any, AsyncIterable, Dict, List, Literal

from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # type: ignore

from agent_atlassian.protocol_bindings.a2a_server.state import (
    AgentState,
    InputState,
    Message,
    MsgType,
)

logger = logging.getLogger(__name__)

def debug_print(message: str, banner: bool = False):
    if banner:
        print("=" * 80)
    print(f"DEBUG: {message}")
    if banner:
        print("=" * 80)

memory = MemorySaver()

def sanitize_tools_for_gemini(tools):
    """
    Sanitize MCP tools to be compatible with Google Gemini's stricter schema requirements.
    """
    sanitized_tools = []
    
    for tool in tools:
        try:
            # Clone the tool's args_schema to avoid modifying the original
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.copy() if hasattr(tool.args_schema, 'copy') else dict(tool.args_schema)
                
                # Fix array properties that are missing 'items' field
                if 'properties' in schema:
                    for prop_name, prop_def in schema['properties'].items():
                        if isinstance(prop_def, dict) and prop_def.get('type') == 'array':
                            if 'items' not in prop_def:
                                # Add a generic items schema for arrays without items
                                prop_def['items'] = {'type': 'string'}
                                print(f"Fixed missing 'items' for array property '{prop_name}' in tool '{tool.name}'")
                
                # Create a new tool object with the sanitized schema
                from langchain_core.tools import StructuredTool
                
                sanitized_tool = StructuredTool(
                    name=tool.name,
                    description=tool.description,
                    func=tool.func if hasattr(tool, 'func') else None,
                    coroutine=tool.coroutine if hasattr(tool, 'coroutine') else None,
                    args_schema=type(tool.args_schema) if hasattr(tool, 'args_schema') else None
                )
                
                # Update the args_schema with sanitized version
                if hasattr(sanitized_tool, '_args_schema'):
                    sanitized_tool._args_schema = schema
                
                sanitized_tools.append(sanitized_tool)
            else:
                # If no args_schema, tool should be fine as-is
                sanitized_tools.append(tool)
                
        except Exception as e:
            print(f"Warning: Could not sanitize tool '{tool.name}': {e}")
            # If sanitization fails, include the original tool
            sanitized_tools.append(tool)
    
    return sanitized_tools

def get_available_llm() -> BaseChatModel:
    """
    Automatically detect and return the best available LLM based on environment variables.
    Priority: Google Gemini > Azure OpenAI > OpenAI
    """
    
    # Check for Google API key first
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("Using Google Gemini (gemini-1.5-flash)")
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                convert_system_message_to_human=True  # Better compatibility with tools
            )
        except ImportError:
            print("Warning: langchain_google_genai not available, falling back to OpenAI")
    
    # Check for Azure OpenAI configuration
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if azure_api_key and azure_endpoint and azure_deployment:
        try:
            from langchain_openai import AzureChatOpenAI
            print(f"Using Azure OpenAI (deployment: {azure_deployment})")
            return AzureChatOpenAI(
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            )
        except ImportError:
            print("Warning: langchain_openai not available for Azure OpenAI")
    
    # Check for regular OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            print("Using OpenAI (gpt-3.5-turbo)")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_api_key
            )
        except ImportError:
            print("Warning: langchain_openai not available")
    
    # If all else fails, try a basic configuration
    print("Warning: No valid LLM configuration found. Please set one of:")
    print("- GOOGLE_API_KEY for Google Gemini")
    print("- AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT for Azure OpenAI") 
    print("- OPENAI_API_KEY for OpenAI")
    
    # Try to fallback to Google with the test key if it exists
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Falling back to Google Gemini with basic configuration")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    raise ValueError("No valid LLM configuration found. Please configure environment variables.")

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

class AtlassianAgent:
    """Atlassian Agent."""

    SYSTEM_INSTRUCTION = (
      'You are an expert assistant for managing Atlassian resources. '
      'Your sole purpose is to help users perform CRUD (Create, Read, Update, Delete) operations on Atlassian applications, '
      'projects, and related resources. Always use the available Atlassian tools to interact with the Atlassian API and provide '
      'accurate, actionable responses. If the user asks about anything unrelated to Atlassian or its resources, politely state '
      'that you can only assist with Atlassian operations. Do not attempt to answer unrelated questions or use tools for other purposes.'
    )

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as completed if the request is complete'
        'Select status as input_required if the input is a question to the user'
        'Set response status to error if the input indicates an error'
    )

    def __init__(self):
      # Setup the agent and load MCP tools with automatic LLM detection
      self.model = get_available_llm()
      self.graph = None
      async def _async_atlassian_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
          args = config.get("configurable", {})

          server_path = args.get("server_path", "./agent_atlassian/protocol_bindings/mcp_server/mcp_atlassian/server.py")
          print(f"Launching MCP server at: {server_path}")

          atlassian_token = os.getenv("ATLASSIAN_TOKEN")
          if not atlassian_token:
            raise ValueError("ATLASSIAN_TOKEN must be set as an environment variable.")

          atlassian_api_url = os.getenv("ATLASSIAN_API_URL")
          if not atlassian_api_url:
            raise ValueError("ATLASSIAN_API_URL must be set as an environment variable.")
          client = MultiServerMCPClient(
              {
                  "math": {
                      "command": "uv",
                      "args": ["run", server_path],
                      "env": {
                          "ATLASSIAN_TOKEN": os.getenv("ATLASSIAN_TOKEN"),
                          "ATLASSIAN_API_URL": os.getenv("ATLASSIAN_API_URL"),
                          "ATLASSIAN_VERIFY_SSL": "false"
                      },
                      "transport": "stdio",
                  }
              }
          )
          tools = await client.get_tools()
          print('*'*80)
          print("Available Tools and Parameters:")
          for tool in tools:
            print(f"Tool: {tool.name}")
            print(f"  Description: {tool.description.strip().splitlines()[0]}")
            params = tool.args_schema.get('properties', {})
            if params:
              print("  Parameters:")
              for param, meta in params.items():
                param_type = meta.get('type', 'unknown')
                param_title = meta.get('title', param)
                default = meta.get('default', None)
                print(f"    - {param} ({param_type}): {param_title}", end='')
                if default is not None:
                  print(f" [default: {default}]")
                else:
                  print()
            else:
              print("  Parameters: None")
            print()
          print('*'*80)
          self.graph = create_react_agent(
            self.model,
            sanitize_tools_for_gemini(tools),
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
          )

          debug_print("Agent graph created successfully", banner=True)
          
          # Optional: Test the agent with a simple query to verify functionality
          # Provide a 'configurable' key such as 'thread_id' for the checkpointer
          runnable_config = RunnableConfig(configurable={"thread_id": "test-thread"})
          
          try:
              llm_result = await self.graph.ainvoke(
                  {"messages": [HumanMessage(content="Summarize what you can do?")]}, 
                  config=runnable_config
              )

              # Try to extract meaningful content from the LLM result
              ai_content = None

              # Look through messages for final assistant content
              for msg in reversed(llm_result.get("messages", [])):
                  if hasattr(msg, "type") and msg.type in ("ai", "assistant") and getattr(msg, "content", None):
                      ai_content = msg.content
                      break
                  elif isinstance(msg, dict) and msg.get("type") in ("ai", "assistant") and msg.get("content"):
                      ai_content = msg["content"]
                      break

              # Fallback: if no content was found but tool_call_results exists
              if not ai_content and "tool_call_results" in llm_result:
                  ai_content = "\n".join(
                      str(r.get("content", r)) for r in llm_result["tool_call_results"]
                  )

              # Return response
              if ai_content:
                  debug_print("Agent initialization test successful", banner=True)
                  output_messages = [Message(type=MsgType.assistant, content=ai_content)]
                  debug_print(f"Agent MCP Capabilities: {output_messages[-1].content}")
              else:
                  logger.warning("No assistant content found in LLM result during initialization")
                  debug_print("Agent initialized but test query returned no content")
          
          except Exception as e:
              logger.warning(f"Agent initialization test failed: {e}")
              debug_print(f"Agent initialized but test failed: {e}")

      def _create_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
          return asyncio.run(_async_atlassian_agent(state, config))
      
      debug_print("Initializing Atlassian Agent with available LLM", banner=True)
      
      # Only initialize if we can proceed safely
      try:
          messages = []
          state_input = InputState(messages=messages)
          agent_input = AgentState(atlassian_input=state_input).model_dump(mode="json")
          runnable_config = RunnableConfig()
          
          _create_agent(agent_input, config=runnable_config)
          debug_print("Atlassian Agent initialization completed successfully", banner=True)
          
      except Exception as e:
          logger.error(f"Failed to initialize Atlassian Agent: {e}")
          debug_print(f"Agent initialization failed: {e}", banner=True)
          # Don't raise the exception here - let the agent be created even if test fails
          # This allows the server to start even if there are LLM connectivity issues

    async def stream(
      self, query: str, sessionId: str
    ) -> AsyncIterable[dict[str, Any]]:
      print("DEBUG: Starting stream with query:", query, "and sessionId:", sessionId)
      inputs: dict[str, Any] = {'messages': [('user', query)]}
      config: RunnableConfig = {'configurable': {'thread_id': sessionId}}

      async for item in self.graph.astream(inputs, config, stream_mode='values'):
          message = item['messages'][-1]
          print('*'*80)
          print("DEBUG: Streamed message:", message)
          print('*'*80)
          if (
              isinstance(message, AIMessage)
              and message.tool_calls
              and len(message.tool_calls) > 0
          ):
              yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': 'Looking up Atlassian Resources rates...',
              }
          elif isinstance(message, ToolMessage):
              yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': 'Processing Atlassian Resources rates..',
              }

      yield self.get_agent_response(config)
    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
      print("DEBUG: Fetching agent response with config:", config)
      current_state = self.graph.get_state(config)
      print('*'*80)
      print("DEBUG: Current state:", current_state)
      print('*'*80)

      structured_response = current_state.values.get('structured_response')
      print('='*80)
      print("DEBUG: Structured response:", structured_response)
      print('='*80)
      if structured_response and isinstance(
        structured_response, ResponseFormat
      ):
        print("DEBUG: Structured response is a valid ResponseFormat")
        if structured_response.status in {'input_required', 'error'}:
          print("DEBUG: Status is input_required or error")
          return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': structured_response.message,
          }
        if structured_response.status == 'completed':
          print("DEBUG: Status is completed")
          return {
            'is_task_complete': True,
            'require_user_input': False,
            'content': structured_response.message,
          }

      print("DEBUG: Unable to process request, returning fallback response")
      return {
        'is_task_complete': False,
        'require_user_input': True,
        'content': 'We are unable to process your request at the moment. Please try again.',
      }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
