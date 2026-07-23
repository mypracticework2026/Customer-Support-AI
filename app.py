import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─── Load models ──────────────────────────────────────────────
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('logistic_model.pkl')
    print("✅ Models loaded with joblib")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise

# ─── Intent mapping ──────────────────────────────────────────
INTENT_MAPPING = {
    "create_account": {
        "department": "Account Support",
        "priority": "High",
        "action": "Assist customer with account creation."
    },
    "edit_account": {
        "department": "Account Support",
        "priority": "High",
        "action": "Update customer account information."
    },
    "delete_account": {
        "department": "Account Support",
        "priority": "High",
        "action": "Verify identity before deleting the account."
    },
    "switch_account": {
        "department": "Account Support",
        "priority": "High",
        "action": "Help customer switch between accounts."
    },
    "recover_password": {
        "department": "Account Support",
        "priority": "High",
        "action": "Guide customer through password recovery."
    },
    "registration_problems": {
        "department": "Account Support",
        "priority": "High",
        "action": "Resolve registration issues."
    },
    "payment_issue": {
        "department": "Billing Team",
        "priority": "High",
        "action": "Investigate the payment transaction."
    },
    "check_invoice": {
        "department": "Billing Team",
        "priority": "Medium",
        "action": "Provide invoice details."
    },
    "get_invoice": {
        "department": "Billing Team",
        "priority": "Medium",
        "action": "Generate or resend the invoice."
    },
    "get_refund": {
        "department": "Billing Team",
        "priority": "High",
        "action": "Start the refund process."
    },
    "track_refund": {
        "department": "Billing Team",
        "priority": "Medium",
        "action": "Check refund status."
    },
    "check_refund_policy": {
        "department": "Billing Team",
        "priority": "Low",
        "action": "Share the refund policy."
    },
    "check_payment_methods": {
        "department": "Billing Team",
        "priority": "Low",
        "action": "Explain available payment methods."
    },
    "check_cancellation_fee": {
        "department": "Billing Team",
        "priority": "Low",
        "action": "Provide cancellation fee details."
    },
    "place_order": {
        "department": "Order Management",
        "priority": "Medium",
        "action": "Create a new customer order."
    },
    "cancel_order": {
        "department": "Order Management",
        "priority": "High",
        "action": "Verify eligibility and cancel the order."
    },
    "change_order": {
        "department": "Order Management",
        "priority": "Medium",
        "action": "Update the existing order."
    },
    "track_order": {
        "department": "Order Management",
        "priority": "Medium",
        "action": "Provide the latest order tracking status."
    },
    "delivery_period": {
        "department": "Shipping Team",
        "priority": "Medium",
        "action": "Inform the customer about the delivery timeline."
    },
    "delivery_options": {
        "department": "Shipping Team",
        "priority": "Low",
        "action": "Explain available delivery options."
    },
    "set_up_shipping_address": {
        "department": "Shipping Team",
        "priority": "Medium",
        "action": "Add a new shipping address."
    },
    "change_shipping_address": {
        "department": "Shipping Team",
        "priority": "Medium",
        "action": "Update the shipping address before dispatch."
    },
    "contact_customer_service": {
        "department": "Customer Support",
        "priority": "Low",
        "action": "Connect the customer with the support team."
    },
    "contact_human_agent": {
        "department": "Customer Support",
        "priority": "High",
        "action": "Transfer the customer to a live support agent."
    },
    "complaint": {
        "department": "Customer Support",
        "priority": "High",
        "action": "Register the complaint and escalate if necessary."
    },
    "review": {
        "department": "Customer Support",
        "priority": "Low",
        "action": "Record the customer's feedback or review."
    },
    "newsletter_subscription": {
        "department": "Customer Support",
        "priority": "Low",
        "action": "Manage the customer's newsletter subscription."
    }
}

# ─── AI Reply generation ──────────────────────────────────────
# Use the model you specified – gemini-3.5-flash-lite
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Use the model you mentioned – if it doesn't exist, it will fall back to template
    try:
        model_gemini = genai.GenerativeModel('gemini-3.5-flash-lite')
        print("✅ Gemini model 'gemini-3.5-flash-lite' ready")
    except Exception as e:
        print(f"⚠️ Could not load gemini-3.5-flash-lite: {e}")
        model_gemini = None
else:
    model_gemini = None
    print("⚠️ No Gemini API key – using fallback replies")

def generate_ai_reply(query, intent, confidence):
    """Generate a helpful reply using Gemini or fallback template."""
    if model_gemini:
        try:
            prompt = f"""
You are a helpful customer support assistant. 
The customer said: "{query}"
Predicted intent: {intent} (confidence: {confidence:.2f}).
Write a short, empathetic, and helpful reply (max 3 sentences) that addresses their concern.
"""
            response = model_gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini error: {e} – using fallback")
            # fall through to template
    # Fallback template
    friendly_intent = intent.replace('_', ' ').title()
    return f"Thank you for reaching out. Our {INTENT_MAPPING.get(intent, {}).get('department', 'support')} team will assist you with your {friendly_intent} request. We'll get back to you shortly."

# ─── Classes ────────────────────────────────────────────────
classes = [
    'cancel_order', 'change_shipping_address', 'check_cancellation_fee',
    'check_invoice', 'check_payment_methods', 'check_refund_policy',
    'complaint', 'contact_customer_service', 'contact_human_agent',
    'create_account', 'delete_account', 'delivery_options',
    'delivery_period', 'edit_account', 'get_invoice', 'get_refund',
    'newsletter_subscription', 'payment_issue', 'place_order',
    'recover_password', 'registration_problems', 'review',
    'set_up_shipping_address', 'switch_account', 'track_order',
    'track_refund'
]

# ─── Routes ──────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Vectorize and predict
    vec = vectorizer.transform([query])
    proba = model.predict_proba(vec)[0]
    top_idx = np.argmax(proba)
    intent = classes[top_idx]
    confidence = float(proba[top_idx])

    # Get business info
    info = INTENT_MAPPING.get(intent, {
        "department": "General Support",
        "priority": "Medium",
        "action": "Handle customer request."
    })

    # Generate AI reply
    ai_reply = generate_ai_reply(query, intent, confidence)

    # All intents (for the dropdown chips)
    all_intents = [
        {'intent': classes[i], 'confidence': float(proba[i])}
        for i in range(len(classes))
        if proba[i] > 0.01
    ]
    all_intents.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'customer_query': query,
        'intent': intent,
        'confidence': confidence,
        'department': info['department'],
        'priority': info['priority'],
        'action': info['action'],
        'ai_reply': ai_reply,
        'all_intents': all_intents
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
