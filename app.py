import gradio as gr
import asyncio
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.agent import safe_run_agent_streaming, safe_run_agent, clear_memory
    from src.data_loaders import process_uploaded_file
    from src.utils import initialize_knowledge_base
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Initialize knowledge base on startup
logger.info("Initializing knowledge base...")
try:
    knowledge_base = initialize_knowledge_base()
    if knowledge_base:
        logger.info("Knowledge base initialized successfully")
    else:
        logger.warning("Knowledge base initialization failed - some features may be limited")
except Exception as e:
    logger.error(f"Knowledge base initialization error: {e}")
    knowledge_base = None

# Global variable to store processed document context
processed_docs = []

async def chat_function_streaming(message: str, history: list):
    """Process user message through the agent with streaming and better error handling"""
    if not message or not message.strip():
        history.append([message, "عذراً، لم أتلقَ أي سؤال. يرجى إدخال سؤالك أو طلبك."])
        yield history, ""
        return
    
    # Add user message to history with empty response
    history.append([message, ""])
    
    try:
        # Prepare message with document context if available
        message_to_agent = message
        if processed_docs:
            str_processed_docs = "\n".join([
                f"{doc.page_content}\n{doc.metadata}" 
                for doc in processed_docs
            ])
            message_to_agent = f"{message}\n\nThis is Information you can use:\n\n{str_processed_docs}"
        
        # Stream the response
        accumulated_response = ""
        async for chunk in safe_run_agent_streaming(message_to_agent):
            accumulated_response += chunk
            history[-1][1] = accumulated_response
            yield history, ""
            
    except Exception as e:
        logger.error(f"Error in chat_function_streaming: {e}")
        history[-1][1] = f"عذراً، حدث خطأ: {str(e)}"
        yield history, ""

def upload_and_process_file(file) -> str:
    """Process uploaded file and add to knowledge base"""
    global processed_docs
    
    if file is None:
        return "لم يتم رفع أي ملف"
    
    try:
        # Gradio's file object has a .name attribute which is the path to the temporary file
        file_path = Path(file)  # file is already a path string in newer versions
        
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        if file_path.suffix.lower() not in allowed_extensions:
            return f"نوع الملف غير مدعوم: {file_path.suffix}. الأنواع المدعومة: {', '.join(allowed_extensions)}"
        
        # Check file size (limit to 10MB)
        file_size = file_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            return "الملف كبير جداً. الحد الأقصى 10 ميجابايت."
        
        # Process the uploaded file
        new_documents = process_uploaded_file(file_path)
        
        if new_documents:
            # Extend the global processed_docs list with the new documents
            processed_docs.extend(new_documents)
            return f"تم معالجة الملف '{file_path.name}' بنجاح. تمت إضافة {len(new_documents)} وثيقة."
        else:
            return f"لم يتم العثور على محتوى قابل للمعالجة في الملف '{file_path.name}'"
        
    except Exception as e:
        logger.error(f"Error processing file {file}: {e}")
        return f"خطأ في معالجة الملف '{file}': {str(e)}"

def clear_chat_memory_and_history():
    """Clear the conversation memory, processed documents, chat history, and upload status"""
    global processed_docs
    
    try:
        clear_memory()
        processed_docs = []
        logger.info("Successfully cleared chat memory and history")
        # Return empty chat history, clear status, and empty upload status
        return [], "تم مسح الذاكرة بنجاح. بدأت محادثة جديدة!", ""
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return [], f"خطأ في مسح الذاكرة: {str(e)}", ""

def validate_startup():
    """Validate system before launching"""
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Startup validation passed")

# Wrapper function to handle async streaming for Gradio
def chat_function_wrapper(message, history):
    """Wrapper to run the async streaming function"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async generator
            async_gen = chat_function_streaming(message, history)
            
            # Iterate through the async generator
            while True:
                try:
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
                    
        finally:
            loop.close()
            
    except Exception as e:
        # Fallback to non-streaming if streaming fails
        try:
            # Check if we have processed documents
            if processed_docs:
                # Create string representation of processed documents
                str_processed_docs = "\n".join([
                    f"{doc.page_content}\n{doc.metadata}" 
                    for doc in processed_docs
                ])
                message_to_agent = f"{message}\n\nThis is Information you can use:\n\n{str_processed_docs}"
                # Pass message with document context to the agent
                response = asyncio.run(safe_run_agent(message_to_agent))
            else:
                # Process normally without document context
                response = asyncio.run(safe_run_agent(message))
            
            # Add the new conversation to history
            history.append([message, response])
            yield history, ""
        except Exception as fallback_error:
            history.append([message, f"Error: {str(fallback_error)}"])
            yield history, ""

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for full-screen responsive layout
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');

    /* Global font */
    * {
        font-family: 'Roboto', sans-serif;
    }

    /* Better RTL support for Arabic */
    .rtl {
        direction: rtl;
        text-align: right;
    }

    /* Apply Roboto font only to English content explicitly if needed */
    :lang(en) {
        font-family: 'Roboto', sans-serif;
    }

    /* Responsive design for small screens */
    @media (max-width: 768px) {
        .gradio-row {
            flex-direction: column !important;
        }
    }

    /* 👇 Control font size in the input Textbox */
    textarea {
        font-size: 18px !important;
    }

    /* 👇 Control font size in the Chatbot messages */
    .message, .message-user, .message-ai {
        font-size: 18px !important;
        line-height: 1.6;
    }

    /* 👇 Optional: Adjust file upload input and other text areas */
    input[type="file"], .gr-textbox, .gr-textbox textarea {
        font-size: 16px !important;
    }
    """

    
    # Create the Gradio interface with full-screen layout
    with gr.Blocks(
        title="الشفاء الرقمية للرعاية الصحية - المساعد الطبي", 
        css=custom_css,
        theme=gr.themes.Soft()
    ) as interface:
        
        # Arabic Header with RTL support
        gr.Markdown(
            """
            <div style="text-align: center; direction: rtl;">
            
            # 🏥 الشفاء الرقمية للرعاية الصحية - المساعد الطبي
            
            ### اطرح أسئلة طبية أو ارفع وثائق للحصول على مساعدة 
            
            </div>
            """, 
            elem_classes="rtl"
        )
        
        # Full-width responsive layout
        with gr.Row():
            with gr.Column(scale=3, min_width=400):  # Increased scale for chat area
                chatbot = gr.Chatbot(
                    label="💬 محادثة المساعد الطبي",
                    height=600,  # Increased height
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    rtl=True  # Enable RTL for Arabic support
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="رسالتك",
                        placeholder="اطرح سؤالاً طبياً باللغة العربية أو الإنجليزية...",
                        scale=4,
                        container=False,
                        rtl=True
                    )
                    submit_btn = gr.Button("إرسال", variant="primary", scale=1)
                
                gr.Markdown(
                    """
                    <div style="text-align: center; direction: rtl; color: #666; margin-top: 10px;">
                    <em>
                    ملاحظة: هذا مشروع تجريبي شخصي لاغراض تعليمية فقط. 
                    <br>
                    لإنشاء مشروع مماثل، يمكنكم التواصل مع المطور: 
                    <br>
                     <a href="mailto:moazeldsoky8@gmail.com">Email</a> | <a href="https://github.com/MoazEldsouky">GitHub</a> | <a href="https://www.linkedin.com/in/moaz-eldesouky-762288251/">LinkedIn</a>
                    <br>
                     WhatsApp: +201096448317
                    <br>
                    في حالات الطوارئ، اتصل بـ 997 فوراً.
                    </em>
                    </div>
                    """,
                    elem_classes="rtl"
                )
                
            with gr.Column(scale=2, min_width=300):  # Side panel
                # Document Upload Section
                gr.Markdown("### 📁 رفع الوثائق", elem_classes="rtl")
                file_upload = gr.File(
                    label="ارفع وثيقة طبية",
                    file_types=[".pdf", ".txt", ".docx", ".doc"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="حالة الرفع", 
                    interactive=False,
                    max_lines=3,
                    rtl=True
                )
                
                # Memory Management Section
                gr.Markdown("### 🧠 إدارة الذاكرة", elem_classes="rtl")
                clear_btn = gr.Button(
                    "🗑️ مسح الذاكرة وبدء محادثة جديدة", 
                    variant="secondary",
                    size="lg"
                )
                clear_status = gr.Textbox(
                    label="حالة المسح", 
                    interactive=False,
                    rtl=True
                )
                
                # About section in Arabic
                with gr.Accordion("ℹ️ حول مساعد الشفاء الرقمي", open=False):
                    gr.Markdown(
                        """
                        <div style="direction: rtl; text-align: right;">
                        
                        **الميزات:**
                        - معلومات وإرشادات طبية
                        - مساعدة في حجز المواعيد
                        - دعم تحليل الوثائق
                        - دعم ثنائي اللغة (العربية/الإنجليزية)
                        - متاح 24/7 لخدمتكم
                        
                        **مهم:**
                        - هذا ليس بديلاً عن المشورة الطبية المهنية
                        - في حالات الطوارئ، اتصل دائماً بـ 997
                        - الاستجابات مولدة بالذكاء الاصطناعي ويجب التحقق منها مع المختصين الطبيين
                        
                        **معلومات الاتصال:**
                        - الطوارئ: 997
                        - خدمة العملاء: متوفرة على مدار الساعة
                        - الموقع الإلكتروني: www.alshifadigital.com
                        - الهاتف: 9200-000-000 (متوفر خلال ساعات العمل الرسمية)
                        
                        </div>
                        """,
                        elem_classes="rtl"
                    )
        
        # Event handlers with streaming support
        def submit_message(message, history):
            """Handle message submission"""
            if message.strip():
                yield from chat_function_wrapper(message, history)
        
        # Connect the submit events
        msg.submit(
            submit_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # File upload handler
        file_upload.upload(
            upload_and_process_file,
            inputs=file_upload,
            outputs=upload_status
        )
        
        # Clear memory handler - now clears memory, chat history, and upload status
        clear_btn.click(
            clear_chat_memory_and_history,
            inputs=[],
            outputs=[chatbot, clear_status, upload_status]
        )
    
    return interface

def launch_gradio():
    """Launch Gradio with startup validation"""
    try:
        validate_startup()
        logger.info("Starting Gradio interface...")
        
        interface = create_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv("PORT", 7860)),
            share=bool(os.getenv("GRADIO_SHARE", False)),
            debug=bool(os.getenv("DEBUG", False)),
            show_error=True,
            quiet=False,
            inbrowser=True,
            favicon_path=None,
            ssl_verify=False,
            app_kwargs={}
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio: {e}")
        raise

if __name__ == "__main__":
    launch_gradio()