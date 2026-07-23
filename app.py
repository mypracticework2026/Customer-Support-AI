import os
import pickle
import numpy as np
import sys
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Missing file: {e.filename}")
    print("Make sure both .pkl files are in the project root.")
    sys.exit(1)
    
# Map class indices to intent names
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
        if proba[i] > 0.01  # filter out negligible values
    ]
    all_intents.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'intent': top_intent,
        'confidence': confidence,
        'all_intents': all_intents,
        'query': query
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
