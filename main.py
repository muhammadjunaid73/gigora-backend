import os
import json
import itertools
from typing import Any, List
from dotenv import load_dotenv

# .env file load karo - sab se pehle
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase_auth import datetime

# ============================================================
# STRIPE IMPORT (with fallback)
# ============================================================
try:
    import stripe
except ImportError:
    stripe = None

# ============================================================
# SUPABASE IMPORT
# ============================================================
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# ============================================================
# CREATE APP - ONLY ONCE
# ============================================================
app = FastAPI()

# ============================================================
# CORS - SAB SE PEHLE (ROUTES SE UPAR)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://gigora-frontend-six.vercel.app",
        "https://gigora.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================
# SUPABASE INITIALIZATION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected successfully")
    except Exception as e:
        supabase = None
        print(f"⚠️ Supabase connection failed: {e}")
else:
    supabase = None
    print("⚠️ Supabase not configured - Database updates disabled")

# ============================================================
# EMAIL FUNCTION
# ============================================================
import smtplib
from email.mime.text import MIMEText

def send_email(to: str, subject: str, body: str):
    """Send email using SMTP"""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    if not sender_email or not sender_password:
        print("⚠️ Email credentials not set")
        return
    
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"✅ Email sent to {to}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ============================================================
# OPTIONAL MODULES (Model Compare, Stripe Checkout)
# ============================================================
import importlib
import importlib.util

model_compare_router = None
if importlib.util.find_spec("model_compare") is not None:
    module = importlib.import_module("model_compare")
    model_compare_router = getattr(module, "router", None)

stripe_router = None
if importlib.util.find_spec("stripe_checkout") is not None:
    stripe_module = importlib.import_module("stripe_checkout")
    stripe_router = getattr(stripe_module, "router", None)

if model_compare_router is not None:
    app.include_router(model_compare_router, prefix="/api")

if stripe_router is not None:
    app.include_router(stripe_router)

# ============================================================
# IMPORT GOOGLE GEMINI (NEW SDK)
# ============================================================
from google import genai

# ============================================================
# GEMINI API KEY ROTATION
# ============================================================
_raw_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_API_KEYS = [k for k in _raw_keys if k]

if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found. Set at least GEMINI_API_KEY in your .env file."
    )

def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "****"

print(
    f"DEBUG - Loaded {len(GEMINI_API_KEYS)} Gemini key(s): "
    + ", ".join(_mask(k) for k in GEMINI_API_KEYS)
)

_key_cycle = itertools.cycle(GEMINI_API_KEYS)
GEMINI_MODEL_NAME = "gemini-2.0-flash"

def get_next_gemini_client():
    """Returns a new Gemini client with the next API key"""
    next_key = next(_key_cycle)
    return genai.Client(api_key=next_key)

def generate_with_gemini(prompt: str) -> str:
    """Generate content using Gemini and return text response"""
    client = get_next_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt
    )
    return response.text

# ============================================================
# DATA MODELS
# ============================================================
class ProfileRequest(BaseModel):
    profileText: str

class ProposalRequest(BaseModel):
    job_post: str
    platform: str
    skill: str
    tone: str

class ProposalResponse(BaseModel):
    proposal: str
    key_points: List[str]

class SEORequest(BaseModel):
    title: str
    description: str

class CheckoutRequest(BaseModel):
    user_id: str
    email: str

class CancelSubscriptionRequest(BaseModel):
    userId: str

class UpdateSubscriptionRequest(BaseModel):
    userId: str
    priceId: str

# ============================================================
# HOME ROUTE
# ============================================================
@app.get("/")
def home():
    return {"message": "Gigora Backend Sub-Systems Operating Normally."}

# ============================================================
# OPTIONS ROUTE (CORS preflight ke liye)
# ============================================================
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

# ============================================================
# 1. PROFILE ANALYZER ROUTE
# ============================================================
@app.post("/api/profile")
async def analyze_profile(request: ProfileRequest):
    try:
        prompt = f"""
        Analyze the following freelance profile description.
        Provide a score out of 10 (not 100), exactly 3 distinct strengths (What is Good),
        and exactly 3 distinct weaknesses/improvements (What to Improve).
        Format the output strictly as a JSON object like this:
        {{
            "score": 8.5,
            "strengths": ["point 1", "point 2", "point 3"],
            "weaknesses": ["point 1", "point 2", "point 3"]
        }}
        Profile Text: {request.profileText}
        """
        response_text = generate_with_gemini(prompt)
        result = json.loads(
            response_text.strip().replace("```json", "").replace("```", "")
        )

        raw_score = result.get("score", 7)
        score = max(0, min(10, round(float(raw_score), 1)))

        return {
            "score": score,
            "good": " ".join(result.get("strengths", [])) or "Clear technical skills presentation.",
            "improve": " ".join(result.get("weaknesses", [])) or "Needs stronger call-to-action and quantitative metrics.",
        }
    except Exception as e:
        print("GEMINI ERROR:", e)
        return {
            "score": 7,
            "good": "Clear technical skills presentation. Professional layout logic. Good domain tracking.",
            "improve": "Needs stronger call-to-action. Missing quantitative metrics. SEO keywords integration could be improved.",
        }

# ============================================================
# 2. PROPOSAL GENERATOR ROUTE
# ============================================================
@app.post("/api/proposal")
async def generate_proposal(request: ProposalRequest):
    try:
        prompt = (
            f"Write a winning, short, {request.tone.lower()} freelance "
            f"proposal for this {request.platform} job, highlighting "
            f"{request.skill} expertise: {request.job_post}"
        )
        response_text = generate_with_gemini(prompt)
        return {"proposal": response_text}
    except Exception as e:
        print("GEMINI ERROR:", e)
        return {
            "proposal": (
                "Hi there!\n\nI reviewed your project description and I am "
                "confident in providing optimized code solutions tailored "
                "specifically to your roadmap.\n\nBest regards,\nMuhammad Junaid"
            )
        }

# ============================================================
# 3. GIG SEO ROUTE
# ============================================================
@app.post("/api/seo")
async def optimize_seo(request: SEORequest):
    try:
        prompt = f"""
        Optimize this gig for SEO. Title: {request.title}
        Description: {request.description}
        Return STRICTLY a JSON object in this exact shape (no extra text):
        {{
            "optimized_title": "a punchy, keyword-rich gig title",
            "optimized_description": "a refactored, SEO-friendly description",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
            "tips": ["tip 1", "tip 2", "tip 3"]
        }}
        """
        response_text = generate_with_gemini(prompt)
        result = json.loads(
            response_text.strip().replace("```json", "").replace("```", "")
        )

        return {
            "optimized_title": result.get("optimized_title", request.title),
            "optimized_description": result.get(
                "optimized_description", request.description
            ),
            "tags": result.get("tags", []),
            "tips": result.get("tips", []),
        }
    except Exception as e:
        print("GEMINI ERROR:", e)
        return {
            "optimized_title": f"{request.title} | Expert Service",
            "optimized_description": (
                "Optimized Keywords Found: React, Tailwind CSS, Full-Stack "
                "Developer, Responsive UI.\n\nYour description has been "
                "refactored successfully."
            ),
            "tags": ["React", "Tailwind CSS", "Full-Stack", "Responsive UI", "Web Dev"],
            "tips": [
                "Add measurable results (e.g. '30% faster load times') to your description.",
                "Use all 5 tag slots — partial tag lists rank lower.",
                "Keep your title under 60 characters for full visibility.",
            ],
        }

# ============================================================
# 4. STRIPE CHECKOUT ROUTE
# ============================================================
@app.post("/api/checkout/create-session")
async def create_checkout_session(request: CheckoutRequest):
    if stripe is None:
        raise HTTPException(
            status_code=500, 
            detail="Stripe library not installed. Run: pip install stripe"
        )
    
    try:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        PRICE_ID = os.getenv("STRIPE_PRICE_ID")
        
        if not PRICE_ID:
            raise HTTPException(status_code=400, detail="STRIPE_PRICE_ID not set in .env")
        
        if not stripe.api_key:
            raise HTTPException(status_code=400, detail="STRIPE_SECRET_KEY not set in .env")
        
        print(f"🔑 Creating checkout for user: {request.user_id}, email: {request.email}")
        print(f"💰 Using Price ID: {PRICE_ID}")
        
        # Determine success and cancel URLs based on environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": PRICE_ID,
                    "quantity": 1,
                }
            ],
            mode="subscription",
           success_url=f"{frontend_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
           cancel_url=f"{frontend_url}/payment/cancel",
            client_reference_id=request.user_id,
            customer_email=request.email,
            metadata={
                "user_id": request.user_id
            }
        )
        
        print(f"✅ Checkout session created: {checkout_session.id}")
        return {"id": checkout_session.id, "url": checkout_session.url}
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Stripe Error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)



        # ============================================================
# 4.5 VERIFY CHECKOUT SESSION ROUTE
# ============================================================
@app.get("/api/checkout/verify-session")
async def verify_checkout_session(session_id: str):
    if stripe is None:
        raise HTTPException(
            status_code=500,
            detail="Stripe library not installed. Run: pip install stripe"
        )

    try:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

        session = stripe.checkout.Session.retrieve(session_id)

        payment_status = getattr(session, "payment_status", None)
        is_paid = payment_status == "paid"

        print(f"🔍 Verifying session {session_id}: payment_status={payment_status}")

        return {
            "paid": is_paid,
            "payment_status": payment_status,
            "session_id": session_id,
        }

    except stripe.error.InvalidRequestError as e:
        print(f"❌ Invalid session_id: {e}")
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        print(f"❌ Verify session error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# 5. STRIPE WEBHOOK ROUTE (WITH DATABASE UPDATE)
# ============================================================
@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    if stripe is None:
        raise HTTPException(
            status_code=500, 
            detail="Stripe library not installed. Run: pip install stripe"
        )
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    if not webhook_secret:
        print("⚠️ WEBHOOK SECRET NOT SET in .env")
        raise HTTPException(status_code=400, detail="Webhook secret not configured")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        
        print(f"📩 Webhook received: {event['type']}")
        
        # Handle subscription events
        if event["type"] == "checkout.session.completed":
            # ✅ Get session data with multiple fallbacks
            session = event.data.object
            
            # ✅ Try multiple ways to get user_id
            user_id = None
            customer_email = None
            customer_id = None
            subscription_id = None
            
            # Method 1: Direct attribute access
            user_id = getattr(session, 'client_reference_id', None)
            customer_email = getattr(session, 'customer_email', None)
            customer_id = getattr(session, 'customer', None)
            subscription_id = getattr(session, 'subscription', None)
            
            # Method 2: If direct access failed, try dict conversion
            if user_id is None:
                try:
                    session_dict = session.to_dict_recursive() if hasattr(session, 'to_dict_recursive') else {}
                    user_id = session_dict.get('client_reference_id')
                    if not customer_email:
                        customer_email = session_dict.get('customer_email')
                    if not customer_id:
                        customer_id = session_dict.get('customer')
                    if not subscription_id:
                        subscription_id = session_dict.get('subscription')
                except:
                    pass
            
            # Method 3: Try metadata fallback
            if user_id is None:
                try:
                    metadata = getattr(session, 'metadata', {})
                    if metadata:
                        user_id = metadata.get('user_id')
                    else:
                        # Try to get from session dict
                        session_dict = session.to_dict_recursive() if hasattr(session, 'to_dict_recursive') else {}
                        metadata = session_dict.get('metadata', {})
                        user_id = metadata.get('user_id')
                except:
                    pass
            
            # Method 4: If still None, try to get from client_reference_id attribute
            if user_id is None:
                try:
                    user_id = getattr(session, 'client_reference_id', None)
                except:
                    pass
            
            print(f"✅ User {user_id} ({customer_email}) subscribed successfully!")
            print(f"🔍 Customer ID: {customer_id}")
            print(f"🔍 Subscription ID: {subscription_id}")
            
            # ✅ UPDATE DATABASE - Set user plan to "pro"
            if supabase and user_id:
                try:
                    # Get subscription details for next billing date
                    if subscription_id:
                        subscription_obj = stripe.Subscription.retrieve(subscription_id)
                        # ✅ Convert to dict safely
                        subscription_dict = {}
                        try:
                            subscription_dict = subscription_obj.to_dict_recursive() if hasattr(subscription_obj, 'to_dict_recursive') else {}
                        except:
                            subscription_dict = vars(subscription_obj) if hasattr(subscription_obj, "__dict__") else {}
                        
                        current_period_end = subscription_dict.get('current_period_end')
                        if not current_period_end:
                            current_period_end = getattr(subscription_obj, 'current_period_end', None)
                        
                        if current_period_end:
                            from datetime import datetime
                            next_billing_date = datetime.fromtimestamp(current_period_end).strftime("%Y-%m-%d")
                        else:
                            next_billing_date = "2026-09-04"
                    else:
                        next_billing_date = "2026-09-04"
                    
                    # ✅ Update profile with all fields
                    update_data = {
                        "plan": "pro",
                        "stripe_customer_id": customer_id,
                        "subscription_id": subscription_id,
                        "next_billing_date": next_billing_date
                    }
                    
                    # ✅ Add subscription_item_id if available
                    if subscription_id:
                        try:
                            sub_obj = stripe.Subscription.retrieve(subscription_id)
                            items = getattr(sub_obj, 'items', None)
                            if items and hasattr(items, 'data') and items.data:
                                subscription_item_id = getattr(items.data[0], 'id', None)
                                if subscription_item_id:
                                    update_data["subscription_item_id"] = subscription_item_id
                        except:
                            pass
                    
                    result = supabase.table("profiles").update(update_data).eq("id", user_id).execute()
                    
                    print(f"✅ Database updated for user: {user_id}")
                    print(f"📊 Update result: {result.data}")
                    
                except Exception as e:
                    print(f"❌ Database update failed: {e}")
            else:
                if not supabase:
                    print("⚠️ Supabase not configured - database update skipped")
                if not user_id:
                    print("⚠️ No user_id found - database update skipped")
            
        elif event["type"] == "customer.subscription.created":
            # ✅ Get subscription data
            subscription = event.data.object
            
            # ✅ Get user_id from metadata
            user_id = None
            try:
                # Method 1: Direct metadata access
                metadata = getattr(subscription, 'metadata', {})
                if metadata:
                    user_id = metadata.get('user_id')
                
                # Method 2: If not found, try dict
                if not user_id:
                    sub_dict = subscription.to_dict_recursive() if hasattr(subscription, 'to_dict_recursive') else {}
                    metadata = sub_dict.get('metadata', {})
                    user_id = metadata.get('user_id')
            except:
                pass
            
            print(f"✅ Subscription created for user: {user_id}")
            
            # ✅ UPDATE DATABASE - Ensure user plan is "pro"
            if supabase and user_id:
                try:
                    # Get customer and subscription IDs
                    customer_id = getattr(subscription, 'customer', None)
                    subscription_id = getattr(subscription, 'id', None)
                    
                    update_data = {
                        "plan": "pro",
                        "stripe_customer_id": customer_id,
                        "subscription_id": subscription_id
                    }
                    
                    result = supabase.table("profiles").update(update_data).eq("id", user_id).execute()
                    print(f"✅ Database updated for user: {user_id}")
                except Exception as e:
                    print(f"❌ Database update failed: {e}")
            
        elif event["type"] == "customer.subscription.deleted":
            # ✅ Get subscription data
            subscription = event.data.object
            
            # ✅ Get user_id from metadata
            user_id = None
            try:
                metadata = getattr(subscription, 'metadata', {})
                if metadata:
                    user_id = metadata.get('user_id')
                if not user_id:
                    sub_dict = subscription.to_dict_recursive() if hasattr(subscription, 'to_dict_recursive') else {}
                    metadata = sub_dict.get('metadata', {})
                    user_id = metadata.get('user_id')
            except:
                pass
            
            print(f"❌ User {user_id} cancelled subscription")
            
            # ✅ UPDATE DATABASE - Set user plan to "free"
            if supabase and user_id:
                try:
                    result = supabase.table("profiles").update({
                        "plan": "free",
                        "subscription_id": None,
                        "next_billing_date": None,
                        "subscription_item_id": None
                    }).eq("id", user_id).execute()
                    print(f"✅ Database updated - user {user_id} downgraded to free")
                except Exception as e:
                    print(f"❌ Database update failed: {e}")
            
        elif event["type"] == "invoice.paid":
            invoice = event.data.object
            invoice_dict = {}
            try:
                invoice_dict = invoice.to_dict_recursive() if hasattr(invoice, 'to_dict_recursive') else {}
            except:
                invoice_dict = vars(invoice) if hasattr(invoice, "__dict__") else {}
            
            invoice_id = invoice_dict.get('id') or getattr(invoice, 'id', None)
            customer = invoice_dict.get('customer') or getattr(invoice, 'customer', None)
            subscription = invoice_dict.get('subscription') or getattr(invoice, 'subscription', None)
            
            print(f"💰 Invoice paid: {invoice_id}")
            print(f"💰 Customer: {customer}")
            print(f"💰 Subscription: {subscription}")
            
        return {"status": "success"}
        
    except stripe.error.SignatureVerificationError as e:
        print(f"❌ Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# 6. CANCEL SUBSCRIPTION ROUTE
# ============================================================
@app.post("/api/billing/cancel-subscription")
async def cancel_subscription(request: CancelSubscriptionRequest):
    try:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        user_id = request.userId
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Get subscription from database
        profile = supabase.table("profiles").select("subscription_id").eq("id", user_id).execute()
        
        if not profile.data or len(profile.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        subscription_id = profile.data[0].get("subscription_id")
        
        if not subscription_id:
            raise HTTPException(status_code=400, detail="No active subscription")
        
        # Cancel in Stripe
        if stripe:
            stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            print(f"✅ Subscription cancelled for user: {user_id}")
            
            # Update database
            supabase.table("profiles").update({
                "plan": "free",
                "subscription_id": None,
                "next_billing_date": None
            }).eq("id", user_id).execute()
            
            return {"status": "cancelled", "subscription_id": subscription_id}
        else:
            raise HTTPException(status_code=500, detail="Stripe not configured")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Cancel subscription error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# 7. UPDATE SUBSCRIPTION ROUTE (Upgrade/Downgrade)
# ============================================================
@app.post("/api/billing/update-subscription")
async def update_subscription(request: UpdateSubscriptionRequest):
    try:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        user_id = request.userId
        new_price_id = request.priceId
        
        if not user_id or not new_price_id:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Get current subscription
        profile = supabase.table("profiles").select("subscription_id, subscription_item_id").eq("id", user_id).execute()
        
        if not profile.data or len(profile.data) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        subscription_id = profile.data[0].get("subscription_id")
        subscription_item_id = profile.data[0].get("subscription_item_id")
        
        if not subscription_id or not subscription_item_id:
            raise HTTPException(status_code=400, detail="No active subscription found")
        
        # Update subscription in Stripe
        if stripe:
            stripe.Subscription.modify(
                subscription_id,
                items=[{"id": subscription_item_id, "price": new_price_id}]
            )
            print(f"✅ Subscription updated for user: {user_id}")
            return {"status": "updated", "subscription_id": subscription_id}
        else:
            raise HTTPException(status_code=500, detail="Stripe not configured")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Update subscription error: {e}")
        raise HTTPException(status_code=400, detail=str(e))