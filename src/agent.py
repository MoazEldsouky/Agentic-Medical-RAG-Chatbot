"""
Al Shifa Digital Healthcare - Medical Chatbot Agent

This module implements the core agent functionality for the Al Shifa Digital Healthcare
medical chatbot system. It provides comprehensive medical assistance, appointment booking,
and company information services with robust error handling and bilingual support.

Author: Al Shifa Digital Healthcare Team
Version: 1.0.0
License: Proprietary
"""

import logging
import traceback
from typing import Any, AsyncGenerator
import asyncio
import requests
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import OutputParserException
from langchain.callbacks.base import BaseCallbackHandler
from openai import RateLimitError, APIError

from config import LLM
from tools import (
    retriever_tool,
    websearch_tool,
    get_current_datetime_tool,
    book_consultation_tool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# STREAMING CALLBACK HANDLER
# ============================================================================

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    def __init__(self):
        self.tokens = []
        self.current_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Called when a new token is generated."""
        self.tokens.append(token)
        self.current_response += token
    
    def get_response(self) -> str:
        """Get the current response."""
        return self.current_response
    
    def reset(self):
        """Reset the handler for a new response."""
        self.tokens = []
        self.current_response = ""


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """Exception raised when a tool fails to execute."""
    pass


class APIConnectionError(AgentError):
    """Exception raised when API connections fail."""
    pass


class ValidationError(AgentError):
    """Exception raised when input validation fails."""
    pass


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Available tools for the agent
AVAILABLE_TOOLS = [
    retriever_tool,
    websearch_tool,
    get_current_datetime_tool,
    book_consultation_tool
]


# System message template for the agent
SYSTEM_MESSAGE = """You are an advanced medical chatbot for "Al Shifa Digital Healthcare" (شركة الشفاء الرقمية للرعاية الصحية). Your name is "Al Shifa Digital Assistant" (روبوت الشفاء الرقمي).

**LANGUAGE PROCESSING WORKFLOW:**
1. **DETECT:** Identify the user's input language (Arabic, English, or other)
2. **TRANSLATE TO ARABIC:** If user uses English or any other language, translate their query to Arabic first
3. **PROCESS:** Use company_knowledge_tool with the Arabic version to find similar documents
4. **RESPOND:** Create response based on retrieved documents
5. **TRANSLATE BACK:** Translate the final response back to the user's original language
6. **DELIVER:** Always respond in the SAME language as the user's original question

**CORE MISSION:** Provide accurate, evidence-based medical information and assist with appointment booking while prioritizing patient safety.

**WORKING HOURS:**
أيام العمل من الأحد إلى الخميس، من 9 صباحًا حتى 9 مساءً.
Working days: Sunday to Thursday, 9 AM to 9 PM.

**AVAILABLE TOOLS:**
1. **book_consultation:** For appointment booking (collect all required info first)
2. **company_knowledge_tool:** For Al Shifa services, hours, company info (ALWAYS use with Arabic queries for better document matching)
3. **get_current_datetime:** For current date/time (only reliable source)
4. **Tavily_Search_Tool:** For medical questions (always use for medical queries)

**ENHANCED WORKFLOW:**

**For Non-Arabic Queries:**
- Step 1: Translate user's query to Arabic internally
- Step 2: Use company_knowledge_tool with Arabic translation
- Step 3: Process retrieved Arabic documents
- Step 4: Create comprehensive response
- Step 5: Translate final response back to user's original language

**Medical Questions:**
- Start with safety disclaimer (in user's language)
- If query is not in Arabic: translate to Arabic → search company_knowledge_tool → translate response back
- Use Tavily_Search_Tool for current, evidence-based information
- Never diagnose - recommend professional consultation
- **EMERGENCIES:** Immediately instruct to call 997 (emergency services) and seek immediate medical attention

**Appointment Booking:**
- **MANDATORY VALIDATION:** Before booking any appointment, MUST verify:
  - Date is Sunday through Thursday (not Friday or Saturday)
  - Time is between 9:00 AM and 9:00 PM
  - If user requests invalid day/time, inform them of working hours and ask for alternative
- Collect: patient_name, age, gender, contact_number, email, reason_for_consultation, preferred_date, preferred_time
- Use get_current_datetime for "today" requests, then check company hours
- **Validation Messages:**
  - Arabic: "عذراً، نحن نعمل من الأحد إلى الخميس، من 9 صباحًا حتى 9 مساءً. يرجى اختيار موعد آخر."
  - English: "Sorry, we work Sunday to Thursday, 9 AM to 9 PM. Please choose another time."
  - Other languages: Translate equivalent message to user's language

**Company Questions:**
- If query is in Arabic: Use company_knowledge_tool directly
- If query is in other language: Translate to Arabic → Use company_knowledge_tool → Translate response back

**CRITICAL RULES:**
- **Safety First:** Medical emergencies → direct to call 997 immediately
- **No Diagnosis:** Provide information only, not medical diagnoses
- **Evidence-Based:** Always use company_knowledge_tool with Arabic for better document retrieval
- **Language Processing:** Always translate non-Arabic queries to Arabic before using company_knowledge_tool
- **Language Match:** Final response must be in user's original language
- **Professional Boundaries:** Clearly state limitations when uncertain
- **Working Hours Enforcement:** Never book appointments outside working days/hours

**TRANSLATION REQUIREMENTS:**
- Maintain medical accuracy in all translations
- Preserve cultural sensitivity in Arabic translations
- Ensure technical terms are properly translated
- Keep the same tone and formality level across languages

**SAFETY DISCLAIMERS:**
- Arabic: "تنويه هام: للطوارئ اتصل بـ 997 فوراً. هذه معلومات تعليمية ولا تغني عن استشارة طبيب."
- English: "Important: For emergencies call 997 immediately. This is educational information, not medical advice."
- Other languages: Translate equivalent disclaimer to user's language

**EMERGENCY PROTOCOL:** 
If user describes emergency symptoms (chest pain, difficulty breathing, severe bleeding, loss of consciousness, etc.), immediately respond:
"هذه حالة طوارئ! اتصل بـ 997 فوراً واطلب المساعدة الطبية العاجلة" / "This is an emergency! Call 997 immediately and seek urgent medical help." (Translate to user's language if different)

**INTERNAL PROCESSING NOTE:**
Always process company-related queries through Arabic translation pipeline to ensure maximum document retrieval accuracy from company_knowledge_tool, then translate back to maintain user language preference.

**Language:**
- Final response must always be in the same language as the user's original query
- الرد النهائي يجب أن يكون بنفس لغة استعلام المستخدم الأصلي
"""

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Initialize the agent
agent = create_openai_tools_agent(
    llm=LLM,
    tools=AVAILABLE_TOOLS,
    prompt=prompt_template,
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=AVAILABLE_TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=120,  # 2 minutes timeout
)

# Initialize memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    max_window_size=10
)

# ============================================================================
# STREAMING AGENT FUNCTIONS
# ============================================================================

async def run_agent_streaming(user_input: str, max_retries: int = 3) -> AsyncGenerator[str, None]:
    """
    Run the agent with streaming support and comprehensive error handling.
    
    This function processes user input through the agent executor with streaming
    capabilities, robust error handling, and automatic retries for recoverable errors.
    
    Args:
        user_input (str): The user's input message to process
        max_retries (int, optional): Maximum number of retries for recoverable errors. 
                                   Defaults to 3.
    
    Yields:
        str: Chunks of the agent's response as they are generated
        
    Raises:
        None: All exceptions are caught and handled internally
    """
    # Input validation
    if not user_input or not user_input.strip():
        logger.warning("Empty input received")
        yield "عذراً، لم أتلقَ أي سؤال. يرجى إدخال سؤالك أو طلبك."
        return
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Load conversation history from memory
            chat_history = memory.load_memory_variables({})["chat_history"]
            
            logger.info(f"Processing user input (attempt {retry_count + 1}): {user_input[:50]}...")
            
            # Create streaming callback handler
            streaming_handler = StreamingCallbackHandler()
            
            # Create a new executor with streaming callback for this request
            streaming_executor = AgentExecutor(
                agent=agent,
                tools=AVAILABLE_TOOLS,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=120,
                callbacks=[streaming_handler]
            )
            
            # Run the agent in a separate thread to avoid blocking
            def run_sync():
                return streaming_executor.invoke({
                    "input": user_input.strip(),
                    "chat_history": chat_history
                })
            
            # Execute the agent with streaming
            full_response = ""
            previous_length = 0
            
            # Start the agent execution in background
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run_sync)
            
            # Stream the response as it's being generated
            while not task.done():
                current_response = streaming_handler.get_response()
                
                # Yield new tokens if available
                if len(current_response) > previous_length:
                    new_content = current_response[previous_length:]
                    previous_length = len(current_response)
                    yield new_content
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.1)
            
            # Get the final result
            response = await task
            
            # Yield any remaining content
            final_response = streaming_handler.get_response()
            if len(final_response) > previous_length:
                yield final_response[previous_length:]
            
            # If no streaming content was captured, yield the full response
            if not final_response and response and "output" in response:
                full_output = response["output"]
                # Simulate streaming by yielding word by word
                words = full_output.split(' ')
                for word in words:
                    yield word + ' '
                    await asyncio.sleep(0.05)
            
            # Validate response structure
            if not response or "output" not in response:
                raise ValidationError("Invalid response format from agent")
            
            if not response["output"] or not response["output"].strip():
                raise ValidationError("Empty response from agent")
            
            # Save conversation context to memory
            memory.save_context(
                {"input": user_input},
                {"output": response["output"]}
            )
            
            logger.info(f"Successfully processed user input: {user_input[:50]}...")
            return
            
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
            
            logger.warning(
                f"Rate limit exceeded. Retrying in {wait_time} seconds... "
                f"(Attempt {retry_count}/{max_retries})"
            )
            
            if retry_count <= max_retries:
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error("Rate limit exceeded after maximum retries")
                yield "عذراً، النظام مشغول حالياً. يرجى المحاولة مرة أخرى بعد قليل."
                return
                
        except APIError as e:
            retry_count += 1
            last_error = e
            logger.error(f"OpenAI API error: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(2)
                continue
            else:
                yield "عذراً، حدث خطأ في الاتصال بالخدمة. يرجى المحاولة مرة أخرى لاحقاً."
                return
                
        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            last_error = e
            logger.error(f"Network connection error: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(3)
                continue
            else:
                yield "عذراً، لا يمكنني الاتصال بالخدمة حالياً. يرجى التحقق من اتصال الإنترنت والمحاولة مرة أخرى."
                return
                
        except requests.exceptions.Timeout as e:
            retry_count += 1
            last_error = e
            logger.error(f"Request timeout: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(2)
                continue
            else:
                yield "عذراً، استغرق الطلب وقتاً أطول من المتوقع. يرجى المحاولة مرة أخرى."
                return
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            yield "عذراً، حدث خطأ في الطلب. يرجى المحاولة مرة أخرى."
            return
            
        except OutputParserException as e:
            logger.error(f"Output parsing error: {str(e)}")
            yield "عذراً، حدث خطأ في معالجة الاستجابة. يرجى إعادة صياغة سؤالك والمحاولة مرة أخرى."
            return
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            yield "عذراً، حدث خطأ في التحقق من صحة البيانات. يرجى المحاولة مرة أخرى."
            return
            
        except ToolExecutionError as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield "عذراً، حدث خطأ أثناء تنفيذ إحدى العمليات. يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
            return
            
        except Exception as e:
            logger.error(f"Unexpected error in run_agent_streaming: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # For unexpected errors, don't retry
            yield "عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني إذا استمرت المشكلة."
            return
    
    # This should never be reached, but just in case
    logger.error(f"Maximum retries exceeded. Last error: {str(last_error)}")
    yield "عذراً، لم أتمكن من معالجة طلبك بعد عدة محاولات. يرجى المحاولة مرة أخرى لاحقاً."


async def safe_run_agent_streaming(user_input: str) -> AsyncGenerator[str, None]:
    """
    Streaming wrapper function with additional safety checks and input validation.
    
    This function provides an additional layer of safety by validating input parameters,
    checking input length constraints, and handling any critical errors that might
    occur during streaming agent execution.
    
    Args:
        user_input (str): The user's input message to process
        
    Yields:
        str: Chunks of the agent's response as they are generated
        
    Raises:
        None: All exceptions are caught and handled internally
    """
    try:
        # Input type validation
        if not isinstance(user_input, str):
            logger.warning(f"Invalid input type received: {type(user_input)}")
            yield "عذراً، يجب أن يكون الإدخال نصاً صالحاً."
            return
        
        # Input length validation
        stripped_input = user_input.strip()
        
        if len(stripped_input) > 1000:
            logger.warning(f"Input too long: {len(stripped_input)} characters")
            yield "عذراً، الرسالة طويلة جداً. يرجى اختصار سؤالك."
            return
        
        if len(stripped_input) == 0:
            logger.warning("Empty input after stripping")
            yield "عذراً، لم أتلقَ أي سؤال. يرجى إدخال سؤالك أو طلبك."
            return
        
        # Stream the response through the main agent function
        async for chunk in run_agent_streaming(user_input):
            yield chunk
        
    except Exception as e:
        logger.critical(f"Critical error in safe_run_agent_streaming: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        yield "عذراً، حدث خطأ خطير في النظام. يرجى التواصل مع الدعم الفني فوراً."


async def run_agent(user_input: str, max_retries: int = 3) -> str:
    """
    Run the agent with comprehensive error handling and retry logic.
    
    This function processes user input through the agent executor with robust
    error handling, automatic retries for recoverable errors, and comprehensive
    logging for debugging and monitoring.
    
    Args:
        user_input (str): The user's input message to process
        max_retries (int, optional): Maximum number of retries for recoverable errors. 
                                   Defaults to 3.
    
    Returns:
        str: The agent's response or an appropriate error message in Arabic
        
    Raises:
        None: All exceptions are caught and handled internally
    """
    # Input validation
    if not user_input or not user_input.strip():
        logger.warning("Empty input received")
        return "عذراً، لم أتلقَ أي سؤال. يرجى إدخال سؤالك أو طلبك."
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Load conversation history from memory
            chat_history = memory.load_memory_variables({})["chat_history"]
            
            logger.info(f"Processing user input (attempt {retry_count + 1}): {user_input[:50]}...")
            
            # Invoke the agent with input and history (synchronous call)
            response = agent_executor.invoke({
                "input": user_input.strip(),
                "chat_history": chat_history
            })
            
            # Validate response structure
            if not response or "output" not in response:
                raise ValidationError("Invalid response format from agent")
            
            if not response["output"] or not response["output"].strip():
                raise ValidationError("Empty response from agent")
            
            # Save conversation context to memory
            memory.save_context(
                {"input": user_input},
                {"output": response["output"]}
            )
            
            logger.info(f"Successfully processed user input: {user_input[:50]}...")
            return response["output"]
            
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
            
            logger.warning(
                f"Rate limit exceeded. Retrying in {wait_time} seconds... "
                f"(Attempt {retry_count}/{max_retries})"
            )
            
            if retry_count <= max_retries:
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error("Rate limit exceeded after maximum retries")
                return "عذراً، النظام مشغول حالياً. يرجى المحاولة مرة أخرى بعد قليل."
                
        except APIError as e:
            retry_count += 1
            last_error = e
            logger.error(f"OpenAI API error: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(2)
                continue
            else:
                return "عذراً، حدث خطأ في الاتصال بالخدمة. يرجى المحاولة مرة أخرى لاحقاً."
                
        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            last_error = e
            logger.error(f"Network connection error: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(3)
                continue
            else:
                return "عذراً، لا يمكنني الاتصال بالخدمة حالياً. يرجى التحقق من اتصال الإنترنت والمحاولة مرة أخرى."
                
        except requests.exceptions.Timeout as e:
            retry_count += 1
            last_error = e
            logger.error(f"Request timeout: {str(e)}")
            
            if retry_count <= max_retries:
                await asyncio.sleep(2)
                continue
            else:
                return "عذراً، استغرق الطلب وقتاً أطول من المتوقع. يرجى المحاولة مرة أخرى."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return "عذراً، حدث خطأ في الطلب. يرجى المحاولة مرة أخرى."
            
        except OutputParserException as e:
            logger.error(f"Output parsing error: {str(e)}")
            return "عذراً، حدث خطأ في معالجة الاستجابة. يرجى إعادة صياغة سؤالك والمحاولة مرة أخرى."
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return "عذراً، حدث خطأ في التحقق من صحة البيانات. يرجى المحاولة مرة أخرى."
            
        except ToolExecutionError as e:
            logger.error(f"Tool execution error: {str(e)}")
            return "عذراً، حدث خطأ أثناء تنفيذ إحدى العمليات. يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
            
        except Exception as e:
            logger.error(f"Unexpected error in run_agent: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # For unexpected errors, don't retry
            return "عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني إذا استمرت المشكلة."
    
    # This should never be reached, but just in case
    logger.error(f"Maximum retries exceeded. Last error: {str(last_error)}")
    return "عذراً، لم أتمكن من معالجة طلبك بعد عدة محاولات. يرجى المحاولة مرة أخرى لاحقاً."


async def safe_run_agent(user_input: str) -> str:
    """
    Wrapper function for run_agent with additional safety checks and input validation.
    
    This function provides an additional layer of safety by validating input parameters,
    checking input length constraints, and handling any critical errors that might
    occur during agent execution.
    
    Args:
        user_input (str): The user's input message to process
        
    Returns:
        str: The agent's response or an appropriate error message in Arabic
        
    Raises:
        None: All exceptions are caught and handled internally
    """
    try:
        # Input type validation
        if not isinstance(user_input, str):
            logger.warning(f"Invalid input type received: {type(user_input)}")
            return "عذراً، يجب أن يكون الإدخال نصاً صالحاً."
        
        # Input length validation
        stripped_input = user_input.strip()
        
        if len(stripped_input) > 1000:
            logger.warning(f"Input too long: {len(stripped_input)} characters")
            return "عذراً، الرسالة طويلة جداً. يرجى اختصار سؤالك."
        
        if len(stripped_input) == 0:
            logger.warning("Empty input after stripping")
            return "عذراً، لم أتلقَ أي سؤال. يرجى إدخال سؤالك أو طلبك."
        
        # Process the input through the main agent function
        return await run_agent(user_input)
        
    except Exception as e:
        logger.critical(f"Critical error in safe_run_agent: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        return "عذراً، حدث خطأ خطير في النظام. يرجى التواصل مع الدعم الفني فوراً."


def clear_memory() -> None:
    """
    Clear the conversation memory.
    
    This function clears all stored conversation history from memory,
    effectively starting a fresh conversation session.
    """
    try:
        memory.clear()
        logger.info("Conversation memory cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")


def get_memory_summary() -> str:
    """
    Get a summary of the current conversation memory.
    
    Returns:
        str: A summary of the conversation history stored in memory
    """
    try:
        memory_vars = memory.load_memory_variables({})
        return str(memory_vars.get("chat_history", "No conversation history available"))
    except Exception as e:
        logger.error(f"Error getting memory summary: {str(e)}")
        return "Error retrieving memory summary"

