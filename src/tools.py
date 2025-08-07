from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from retriever import hybrid_retriever
import pytz
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
import json
import uuid


# Optimized retriever tool with faster settings
retriever_tool = create_retriever_tool(
    hybrid_retriever,
    name="company_knowledge_tool",
    description="Fast tool for company info, history, products, and policies. Use only for Al Shifa Digital Healthcare questions."
)

# Optimized web search tool with reduced results for speed
websearch_tool = TavilySearchResults(
    max_results=3,  
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    search_depth="basic",  
    name="Tavily_Search_Tool",
    description="Fast search for medical questions and current info. Use for medical advice and up-to-date information."
)


@tool
def get_current_datetime_tool() -> str:
    """
    Returns the current date, time, and day of the week for Saudi Arabia (Asia/Riyadh).
    This is the only reliable source for date and time information. Use this tool
    whenever a user asks about 'today', 'now', or any other time-sensitive query.
    The output is in English but shows Saudi Arabia local time.
    """
    try:
        # Define the timezone for Saudi Arabia
        saudi_tz = pytz.timezone('Asia/Riyadh')
        
        # Get the current time in that timezone
        now_saudi = datetime.now(saudi_tz)
        
        # Manual mapping to ensure English output regardless of system locale
        days_en = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"
        }
        months_en = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        # Get English names using manual mapping
        day_name = days_en[now_saudi.weekday()]
        month_name = months_en[now_saudi.month]
        day = now_saudi.day
        year = now_saudi.year
        
        # Format time manually to avoid locale issues
        hour = now_saudi.hour
        minute = now_saudi.minute
        
        # Convert to 12-hour format
        if hour == 0:
            hour_12 = 12
            period = "AM"
        elif hour < 12:
            hour_12 = hour
            period = "AM"
        elif hour == 12:
            hour_12 = 12
            period = "PM"
        else:
            hour_12 = hour - 12
            period = "PM"
        
        time_str = f"{hour_12:02d}:{minute:02d} {period}"
        
        # Create the final string
        return f"Current date and time in Saudi Arabia: {day_name}, {month_name} {day}, {year} at {time_str}"
    
    except Exception as e:
        return f"Error getting current datetime: {str(e)}"


# --- Pydantic Model for Structured Tool Input ---
# All fields are now Optional to prevent validation errors at the agent level.
# The validation logic is moved inside the tool itself.
class BookingInput(BaseModel):
    """Inputs for the book_consultation tool."""
    patient_name: Optional[str] = Field(None, description="The user's full name.")
    age: Optional[str] = Field(None, description="The user's age.")
    gender: Optional[str] = Field(None, description="The user's gender.")
    contact_number: Optional[str] = Field(None, description="A phone number for confirmation.")
    email: Optional[str] = Field(None, description="An email address for confirmation and reminders.")
    reason_for_consultation: Optional[str] = Field(None, description="A brief description of the user's symptoms or reason for the visit.")
    preferred_date: Optional[str] = Field(None, description="The user's desired date for the appointment.")
    preferred_time: Optional[str] = Field(None, description="The user's desired time for the appointment.")
    specialty: Optional[str] = Field(None, description="The specific medical field needed.")
    doctor_preference: Optional[str] = Field(None, description="The name of a specific doctor, if requested.")
    consultation_type: Optional[str] = Field(None, description="The method of consultation.")

@tool(args_schema=BookingInput)
def book_consultation_tool(patient_name: Optional[str] = None, age: Optional[str] = None, gender: Optional[str] = None, contact_number: Optional[str] = None, email: Optional[str] = None, reason_for_consultation: Optional[str] = None, preferred_date: Optional[str] = None, preferred_time: Optional[str] = None, consultation_type: Optional[str] = None, specialty: Optional[str] = None, doctor_preference: Optional[str] = None) -> str:
    """
    Books a medical consultation. If required information is missing, it returns a message
    asking for the missing details. Otherwise, it returns the booking data as a JSON object.
    Email is optional and will be set to 'unknown@alshifa-care.com' if not provided.
    """
    # --- 1. Internal Validation: Check for missing required information ---
    missing_fields = []
    if not patient_name:
        missing_fields.append("اسم المريض الكامل (patient_name)")
    if not age:
        missing_fields.append("العمر (age)")
    if not gender:
        missing_fields.append("الجنس (gender)")
    if not contact_number:
        missing_fields.append("رقم التواصل (contact_number)")
    if not reason_for_consultation:
        missing_fields.append("سبب الاستشارة (reason_for_consultation)")
    if not preferred_date:
        missing_fields.append("التاريخ المفضل (preferred_date)")
    if not preferred_time:
        missing_fields.append("الوقت المفضل (preferred_time)")
    # Set default email if not provided
    if not email:
        email = "unknown@alshifa-care.com"
    # --- 2. Handle missing required fields ---
    if missing_fields:
        missing_fields_str = "\n- ".join([""] + missing_fields)  # Add newline and bullet points
        return f"""
        عذراً، لا يمكن إتمام الحجز بسبب نقص المعلومات المطلوبة. يرجى توفير:
        {missing_fields_str}
       
        Sorry, we cannot complete the booking due to missing required information. Please provide:
        {missing_fields_str}
        """
    # --- 2. If all data is present, proceed with booking ---
    booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().isoformat()
    booking_data = {
        "booking_id": booking_id,
        "booking_timestamp": timestamp,
        "patient_details": {"name": patient_name, "age": age, "gender": gender, "contact_number": contact_number, "email": email},
        "consultation_details": {
            "reason": reason_for_consultation,
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "specialty": specialty or "Not Specified",
            "doctor_preference": doctor_preference or "Any",
            "type": consultation_type or "Video Call"
        }
    }
    # 3. Return the final data as a formatted JSON string
    return json.dumps(booking_data, indent=4, ensure_ascii=False)