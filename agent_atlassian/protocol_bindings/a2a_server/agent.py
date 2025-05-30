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
    Return the best available LLM based on LLM_PROVIDER environment variable or auto-detection.
    
    If LLM_PROVIDER is set, use that specific provider:
    - "google" or "gemini": Google Gemini 2.5 Flash
    - "azure" or "azure_openai": Azure OpenAI
    - "openai": OpenAI
    
    If LLM_PROVIDER is not set, auto-detect based on available environment variables.
    Priority: Google Gemini > Azure OpenAI > OpenAI
    """
    
    # Check for explicit provider preference first
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    
    if llm_provider in ["google", "gemini", "google-gemini"]:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                print("Using Google Gemini 2.5 Flash (explicit provider)")
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-preview-05-20",
                    convert_system_message_to_human=True
                )
            except ImportError:
                print("Warning: langchain_google_genai not available")
                raise ValueError("Google Gemini requested but langchain_google_genai not available")
        else:
            raise ValueError("Google Gemini requested but GOOGLE_API_KEY not found")
    
    elif llm_provider in ["azure", "azure_openai", "azure-openai"]:
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        if azure_api_key and azure_endpoint and azure_deployment:
            try:
                from langchain_openai import AzureChatOpenAI
                print(f"Using Azure OpenAI (explicit provider, deployment: {azure_deployment})")
                return AzureChatOpenAI(
                    azure_deployment=azure_deployment,
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
            except ImportError:
                print("Warning: langchain_openai not available for Azure OpenAI")
                raise ValueError("Azure OpenAI requested but langchain_openai not available")
        else:
            missing = []
            if not azure_api_key: missing.append("AZURE_OPENAI_API_KEY")
            if not azure_endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
            if not azure_deployment: missing.append("AZURE_OPENAI_DEPLOYMENT")
            raise ValueError(f"Azure OpenAI requested but missing: {', '.join(missing)}")
    
    elif llm_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                print("Using OpenAI (explicit provider)")
                # Ensure we don't pass any Azure-related environment variables
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_api_key,
                    temperature=0,
                    base_url=None  # Explicitly set to None to avoid Azure endpoint issues
                )
            except ImportError:
                print("Warning: langchain_openai not available")
                raise ValueError("OpenAI requested but langchain_openai not available")
        else:
            raise ValueError("OpenAI requested but OPENAI_API_KEY not found")
    
    elif llm_provider and llm_provider not in ["", "auto"]:
        raise ValueError(f"Unknown LLM_PROVIDER: {llm_provider}. Valid options: google, gemini, google-gemini, azure, azure_openai, azure-openai, openai")
    
    # Auto-detection logic (when LLM_PROVIDER is not set or set to "auto")
    print("Auto-detecting available LLM provider...")
    
    # Check for Google API key first
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("Using Google Gemini 2.5 Flash (auto-detected)")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
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
            print(f"Using Azure OpenAI (auto-detected, deployment: {azure_deployment})")
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
            print("Using OpenAI (auto-detected)")
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=0,
                base_url=None  # Explicitly set to None to avoid Azure endpoint issues
            )
        except ImportError:
            print("Warning: langchain_openai not available")
    
    # If all else fails, provide helpful error message
    print("Warning: No valid LLM configuration found. Please set one of:")
    print("- LLM_PROVIDER=google + GOOGLE_API_KEY for Google Gemini 2.5 Flash")
    print("- LLM_PROVIDER=azure + AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT for Azure OpenAI") 
    print("- LLM_PROVIDER=openai + OPENAI_API_KEY for OpenAI")
    print("- Or set the API keys without LLM_PROVIDER for auto-detection")
    
    # Try to fallback to Google with the test key if it exists
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Falling back to Google Gemini 2.5 Flash with basic configuration")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
    
    raise ValueError("No valid LLM configuration found. Please configure environment variables.")

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

class AtlassianAgent:
    """
    Atlassian Agent with multi-model LLM support.
    
    Supports Google Gemini 2.5 Flash, Azure OpenAI, and OpenAI models.
    Use LLM_PROVIDER environment variable for explicit provider selection,
    or let the agent auto-detect based on available API keys.
    """

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
      # Setup the agent with automatic or explicit LLM provider selection
      self.model = get_available_llm()
      self.graph = None
      self._initialized = False
      
      debug_print("Atlassian Agent created successfully (MCP setup deferred)", banner=True)

    async def _ensure_initialized(self):
        """Ensure the agent is properly initialized with MCP tools (called on first request)"""
        if self._initialized:
            return
            
        debug_print("Initializing MCP client and tools...", banner=True)
        
        try:
            server_path = "./agent_atlassian/protocol_bindings/mcp_server/mcp_atlassian/server.py"
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
                params = tool.args_schema.get('properties', {}) if hasattr(tool.args_schema, 'get') else {}
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
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            debug_print(f"MCP initialization failed: {e}", banner=True)
            # Don't raise - allow agent to work without MCP for basic functionality
            self._initialized = True

    async def stream(
      self, query: str, sessionId: str
    ) -> AsyncIterable[dict[str, Any]]:
      print("DEBUG: Starting stream with query:", query, "and sessionId:", sessionId)
      
      # Ensure the agent is properly initialized before processing requests
      await self._ensure_initialized()
      
      if not self.graph:
          yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': 'Agent initialization failed. Please check the logs and try again.',
          }
          return
      
      try:
          inputs: dict[str, Any] = {'messages': [('user', query)]}
          config: RunnableConfig = {
              'configurable': {'thread_id': sessionId},
              'recursion_limit': 10  # Prevent infinite recursion
          }

          iteration_count = 0
          max_iterations = 15

          async for item in self.graph.astream(inputs, config, stream_mode='values'):
              iteration_count += 1
              if iteration_count > max_iterations:
                  print(f"WARNING: Exceeded maximum iterations ({max_iterations}), stopping")
                  yield {
                      'is_task_complete': True,
                      'require_user_input': False,
                      'content': 'Request took too many iterations to complete. Please try a simpler query.',
                  }
                  return
                  
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
                    'content': 'Looking up Atlassian Resources...',
                  }
              elif isinstance(message, ToolMessage):
                  yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing Atlassian Resources...',
                  }

          yield self.get_agent_response(config)
          
      except RecursionError as e:
          print(f"ERROR: Recursion error occurred: {e}")
          yield {
              'is_task_complete': True,
              'require_user_input': False,
              'content': 'Request caused too many recursive calls. Please try a simpler query.',
          }
      except Exception as e:
          print(f"ERROR: Unexpected error during stream: {e}")
          yield {
              'is_task_complete': True,
              'require_user_input': False,
              'content': f'An error occurred while processing your request: {str(e)}',
          }

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
