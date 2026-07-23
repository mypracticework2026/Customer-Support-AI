import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─── Load models ──────────────────────────────────────────────
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Models loaded with pickle")
    print(f"Classes: {model.classes_}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise

# ─── Intent mapping ──────────────────────────────────────────
INTENT_MAPPING = {
    "create_account": {"department": "Account Support", "priority": "High", "action": "Assist customer with account creation.", "reply": "We'll guide you through the account creation process. Please have your email and a secure password ready. If you need assistance, our team is here to help."},
    "edit_account": {"department": "Account Support", "priority": "High", "action": "Update customer account information.", "reply": "We'll update your account details. Please verify your identity and tell us which information you'd like to change. We'll process your request as soon as possible."},
    "delete_account": {"department": "Account Support", "priority": "High", "action": "Verify identity before deleting the account.", "reply": "We understand you want to delete your account. For security, we'll verify your identity first. Please confirm your decision – we're here to help if you have any concerns."},
    "switch_account": {"department": "Account Support", "priority": "High", "action": "Help customer switch between accounts.", "reply": "We'll help you switch between accounts. Please let us know which account you want to use, and we'll guide you through the process."},
    "recover_password": {"department": "Account Support", "priority": "High", "action": "Guide customer through password recovery.", "reply": "We'll help you reset your password. Check your registered email for a recovery link – follow the instructions to create a new password. If you don't see the email, check your spam folder or let us know."},
    "registration_problems": {"department": "Account Support", "priority": "High", "action": "Resolve registration issues.", "reply": "We'll resolve your registration issue. Please describe the problem you're facing (e.g., error message, missing confirmation email). Our team will assist you promptly."},
    "payment_issue": {"department": "Billing Team", "priority": "High", "action": "Investigate the payment transaction.", "reply": "We'll look into your payment issue. Please provide the transaction ID and the last four digits of your card. We'll investigate and get back to you with a resolution as soon as possible."},
    "check_invoice": {"department": "Billing Team", "priority": "Medium", "action": "Provide invoice details.", "reply": "We'll provide your invoice details. Please share your order number or email address, and we'll send the invoice to your registered email."},
    "get_invoice": {"department": "Billing Team", "priority": "Medium", "action": "Generate or resend the invoice.", "reply": "We'll resend your invoice to your registered email. If you need a different format or have any questions, just let us know."},
    "get_refund": {"department": "Billing Team", "priority": "High", "action": "Start the refund process.", "reply": "We'll initiate your refund. Please allow 5‑7 business days for the amount to appear in your account. We'll keep you updated on the status."},
    "track_refund": {"department": "Billing Team", "priority": "Medium", "action": "Check refund status.", "reply": "We'll check the status of your refund. Please allow some time for processing – we'll update you as soon as we have more information."},
    "check_refund_policy": {"department": "Billing Team", "priority": "Low", "action": "Share the refund policy.", "reply": "Our refund policy allows returns within 30 days of purchase. Items must be unused and in original packaging. For digital products, please see our terms. Let us know if you have more questions."},
    "check_payment_methods": {"department": "Billing Team", "priority": "Low", "action": "Explain available payment methods.", "reply": "We accept major credit cards, PayPal, and bank transfers. For corporate orders, we also offer invoice payment. Visit our payment page for full details."},
    "check_cancellation_fee": {"department": "Billing Team", "priority": "Low", "action": "Provide cancellation fee details.", "reply": "Cancellation fees depend on the product and timing. For orders not yet shipped, there is no fee. For shipped items, a restocking fee may apply. Contact us for a specific calculation."},
    "place_order": {"department": "Order Management", "priority": "Medium", "action": "Create a new customer order.", "reply": "We'll place your order. Please review your cart and confirm the shipping address. We'll send a confirmation email once the order is placed."},
    "cancel_order": {"department": "Order Management", "priority": "High", "action": "Verify eligibility and cancel the order.", "reply": "We'll cancel your order if it hasn't been shipped yet. Please confirm your order number and we'll process the cancellation. If already shipped, you may need to return the item."},
    "change_order": {"department": "Order Management", "priority": "Medium", "action": "Update the existing order.", "reply": "We'll update your order. Please tell us what changes you need (e.g., quantity, variant, shipping address). We'll confirm once the update is applied."},
    "track_order": {"department": "Order Management", "priority": "Medium", "action": "Provide the latest order tracking status.", "reply": "We'll track your order. Please share your order number and we'll provide the current status and estimated delivery date. You can also check your account for live updates."},
    "delivery_period": {"department": "Shipping Team", "priority": "Medium", "action": "Inform the customer about the delivery timeline.", "reply": "Standard delivery takes 3‑5 business days. Express shipping is 1‑2 business days. Delivery times may vary based on your location. We'll send tracking details once the order ships."},
    "delivery_options": {"department": "Shipping Team", "priority": "Low", "action": "Explain available delivery options.", "reply": "We offer standard, express, and same‑day delivery (in select areas). You can choose your preferred option at checkout. Shipping costs depend on the method and destination."},
    "set_up_shipping_address": {"department": "Shipping Team", "priority": "Medium", "action": "Add a new shipping address.", "reply": "We'll add your new shipping address. Please provide the full address, including postal code and contact number. We'll save it to your account for future orders."},
    "change_shipping_address": {"department": "Shipping Team", "priority": "Medium", "action": "Update the shipping address before dispatch.", "reply": "We'll update your shipping address if your order hasn't been dispatched yet. Please confirm the new address and we'll make the change. If already shipped, you may need to contact the courier."},
    "contact_customer_service": {"department": "Customer Support", "priority": "Low", "action": "Connect the customer with the support team.", "reply": "Our customer service team is available 24/7. You can reach us via live chat, email, or phone. We'll connect you to the right person to assist with your query."},
    "contact_human_agent": {"department": "Customer Support", "priority": "High", "action": "Transfer the customer to a live support agent.", "reply": "We'll connect you to a human agent. Please hold – a representative will be with you shortly. If you prefer, you can also request a callback."},
    "complaint": {"department": "Customer Support", "priority": "High", "action": "Register the complaint and escalate if necessary.", "reply": "We've registered your complaint and will investigate thoroughly. Our team will review the issue and reach out with a resolution within 24 hours. We take your feedback seriously."},
    "review": {"department": "Customer Support", "priority": "Low", "action": "Record the customer's feedback or review.", "reply": "Thank you for your feedback. Your review helps us improve our service. We'll record your comments and share them with our team. If you need assistance, don't hesitate to ask."},
    "newsletter_subscription": {"department": "Customer Support", "priority": "Low", "action": "Manage the customer's newsletter subscription.", "reply": "We'll manage your newsletter subscription. You can subscribe, unsubscribe, or update your preferences – just let us know what you'd like to do."}
}

DEFAULT_INFO = {
    "department": "General Support",
    "priority": "Medium",
    "action": "Handle customer request.",
    "reply": "We've received your request and will get back to you shortly. Please provide any additional details that might help us assist you better."
}

# ─── Classes – exactly matches model.classes_ ──────────────
classes = [
    'cancel_order',
    'change_order',
    'change_shipping_address',
    'check_cancellation_fee',
    'check_invoice',
    'check_payment_methods',
    'check_refund_policy',
    'complaint',
    'contact_customer_service',
    'contact_human_agent',
    'create_account',
    'delete_account',
    'delivery_options',
    'delivery_period',
    'edit_account',
    'get_invoice',
    'get_refund',
    'newsletter_subscription',
    'payment_issue',
    'place_order',
    'recover_password',
    'registration_problems',
    'review',
    'set_up_shipping_address',
    'switch_account',
    'track_order',
    'track_refund'
]

# ─── Routes ──────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        vec = vectorizer.transform([query])
        proba = model.predict_proba(vec)[0]
        top_idx = np.argmax(proba)
        intent = classes[top_idx]
        confidence = float(proba[top_idx])

        # ─── Debug logging ────────────────────────────────────
        print(f"Query: {query}")
        print(f"Predicted intent: {intent}, confidence: {confidence:.4f}")
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(classes[i], proba[i]) for i in top3_idx]
        print(f"Top 3 predictions: {top3}")
        # ──────────────────────────────────────────────────────

        info = INTENT_MAPPING.get(intent, DEFAULT_INFO)
        reply = info['reply']

        return jsonify({
            'customer_query': query,
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'department': info['department'],
            'priority': info['priority'],
            'action': info['action'],
            'ai_reply': reply,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
