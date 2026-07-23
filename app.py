import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# ─── Load models with joblib ──────────────────────────────────
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('logistic_model.pkl')
    print("✅ Models loaded successfully with joblib!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    raise

# ─── Intent classes (must match training order) ──────────────
classes = [
    'cancel_order',
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

# ─── Routes ─────────────────────────────────────────────────
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
    top_intent = classes[top_idx]
    confidence = float(proba[top_idx])

    # Build list of all intents with confidence
    all_intents = [
        {'intent': classes[i], 'confidence': float(proba[i])}
        for i in range(len(classes))
        if proba[i] > 0.01
    ]
    all_intents.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'intent': top_intent,
        'confidence': confidence,
        'all_intents': all_intents,
        'query': query
    })

# ─── Run ─────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
