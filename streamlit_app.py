import streamlit as st
import pandas as pd
import numpy as np
import time
import string
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="RupeeGuard", page_icon="🛡️", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

# ============================================================
# DATA + MODEL (cached so this only runs once)
# ============================================================

@st.cache_data
def generate_data():
    np.random.seed(42)
    random.seed(42)
    N = 20000

    def rand_id(prefix, length=14):
        chars = string.ascii_letters + string.digits
        return f"{prefix}_" + "".join(random.choices(chars, k=length))

    def fake_email():
        first = "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
        domains = ["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com"]
        return f"{first}{random.randint(1,999)}@{random.choice(domains)}"

    def fake_phone():
        return "+91" + "".join(random.choices(string.digits, k=10))

    methods = ["card", "netbanking", "wallet", "upi", "emi"]
    method_weights = [0.42, 0.13, 0.10, 0.30, 0.05]

    error_map = {
        "card": [
            ("BAD_REQUEST_ERROR", "payment_authentication", "customer", "incorrect_otp", "Payment processing failed because of incorrect OTP"),
            ("BAD_REQUEST_ERROR", "payment_authorization", "customer", "card_declined", "Card declined by the issuing bank"),
            ("GATEWAY_ERROR", "payment_authorization", "bank", "issuer_unavailable", "Card issuer is currently unavailable"),
            ("BAD_REQUEST_ERROR", "payment_authorization", "customer", "insufficient_funds", "Insufficient funds in the account"),
        ],
        "netbanking": [
            ("GATEWAY_ERROR", "payment_authorization", "bank", "bank_technical_error", "Bank server did not respond in time"),
            ("BAD_REQUEST_ERROR", "payment_authentication", "customer", "session_timed_out", "Customer took too long to authenticate"),
        ],
        "wallet": [
            ("BAD_REQUEST_ERROR", "payment_authorization", "customer", "insufficient_balance", "Insufficient balance in wallet"),
            ("GATEWAY_ERROR", "payment_processing", "gateway", "wallet_service_down", "Wallet service temporarily unavailable"),
        ],
        "upi": [
            ("BAD_REQUEST_ERROR", "payment_authorization", "customer", "payment_declined", "UPI payment declined by customer"),
            ("GATEWAY_ERROR", "payment_authorization", "bank", "upi_collect_expired", "UPI collect request expired before approval"),
            ("BAD_REQUEST_ERROR", "payment_authentication", "customer", "incorrect_pin", "Incorrect UPI PIN entered"),
        ],
        "emi": [
            ("BAD_REQUEST_ERROR", "payment_authorization", "customer", "emi_not_supported", "EMI not supported on this card"),
            ("GATEWAY_ERROR", "payment_authorization", "bank", "issuer_unavailable", "Card issuer is currently unavailable"),
        ],
    }

    base_customers = [f"cust_{i}" for i in range(1, 4001)]
    base_devices = [f"device_{i}" for i in range(1, 3000)]
    fraud_device_pool = base_devices[:25]
    shared_home_devices = base_devices[25:75]

    rows = []
    now = int(time.time())

    for i in range(N):
        is_fraud = np.random.choice([0, 1], p=[0.975, 0.025])
        method = np.random.choice(methods, p=method_weights)
        amount_inr = round(np.random.exponential(1800) + 50, 2)
        amount_paise = int(amount_inr * 100)
        created_at = now - random.randint(0, 30 * 24 * 3600)

        if is_fraud == 1:
            device_id = random.choice(fraud_device_pool) if random.random() < 0.7 else random.choice(base_devices)
            customer_id = random.choice(base_customers)
            attempts_last_hour = np.random.randint(3, 20)
            international = np.random.choice([True, False], p=[0.35, 0.65])
            status = np.random.choice(["captured", "failed"], p=[0.35, 0.65])
        else:
            if random.random() < 0.05:
                device_id = random.choice(shared_home_devices)
            else:
                device_id = random.choice(base_devices)
            customer_id = random.choice(base_customers)
            attempts_last_hour = np.random.randint(1, 5)
            international = np.random.choice([True, False], p=[0.03, 0.97])
            status = np.random.choice(["captured", "failed", "created"], p=[0.80, 0.14, 0.06])

        captured = status == "captured"
        fee = int(amount_paise * 0.02) if captured else 0
        tax = int(fee * 0.18) if captured else 0

        if status == "failed":
            error_code, error_step, error_source, error_reason, error_description = random.choice(error_map[method])
        else:
            error_code = error_step = error_source = error_reason = error_description = None

        rows.append({
            "id": rand_id("pay"), "entity": "payment", "order_id": rand_id("order"),
            "amount": amount_paise, "currency": "INR", "status": status, "method": method,
            "international": international, "captured": captured, "amount_refunded": 0,
            "refund_status": None, "fee": fee, "tax": tax, "email": fake_email(), "contact": fake_phone(),
            "error_code": error_code, "error_description": error_description, "error_source": error_source,
            "error_step": error_step, "error_reason": error_reason, "created_at": created_at,
            "customer_id": customer_id, "device_id": device_id,
            "attempts_last_hour": attempts_last_hour, "is_fraud": is_fraud,
        })

    return pd.DataFrame(rows)


@st.cache_resource
def train_guard(df):
    df = df.copy()
    df["device_customer_count"] = df.groupby("device_id")["customer_id"].transform("nunique")
    method_dummies = pd.get_dummies(df["method"], prefix="method")
    feature_cols = ["amount", "international", "attempts_last_hour", "device_customer_count"] + list(method_dummies.columns)
    features = pd.concat([df[["amount", "international", "attempts_last_hour", "device_customer_count"]], method_dummies], axis=1)
    target = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model, feature_cols, method_dummies.columns.tolist()


RECOVERY_ACTION_MAP = {
    "insufficient_funds": "retry_later_24h", "card_declined": "send_reminder",
    "incorrect_otp": "send_reminder", "expired_card": "send_reminder",
    "bank_technical_error": "retry_immediately", "session_timed_out": "send_reminder",
    "insufficient_balance": "send_reminder", "wallet_service_down": "retry_later_2h",
    "payment_declined": "send_reminder", "upi_collect_expired": "resend_collect_request",
    "incorrect_pin": "send_reminder", "emi_not_supported": "suggest_alt_method",
    "issuer_unavailable": "retry_later_2h",
}
ACTION_DISPLAY = {
    "retry_later_24h": "Retry in 24 hours", "send_reminder": "Send a reminder",
    "retry_immediately": "Retry immediately", "retry_later_2h": "Retry in 2 hours",
    "resend_collect_request": "Resend UPI collect request", "suggest_alt_method": "Suggest alternate payment method",
    "escalate_to_human": "Escalate to human review",
}
GUARD_THRESHOLD = 0.70
MAX_ATTEMPTS_BEFORE_ESCALATION = 4


def process_transaction(txn, model, feature_cols, method_cols):
    row = {"amount": txn["amount"], "international": txn["international"],
           "attempts_last_hour": txn["attempts_last_hour"], "device_customer_count": txn["device_customer_count"]}
    for col in method_cols:
        row[col] = (col == f"method_{txn['method']}")
    X_new = pd.DataFrame([row])[feature_cols]
    fraud_prob = float(model.predict_proba(X_new)[0, 1])

    if fraud_prob >= GUARD_THRESHOLD:
        return {"decision": "BLOCKED", "fraud_probability": fraud_prob, "recovery_action": None,
                "reasoning": f"Flagged as fraud with {fraud_prob*100:.1f}% confidence. No recovery attempted."}

    if txn["attempts_last_hour"] >= MAX_ATTEMPTS_BEFORE_ESCALATION:
        return {"decision": "ESCALATED", "fraud_probability": fraud_prob, "recovery_action": "escalate_to_human",
                "reasoning": f"Already failed {txn['attempts_last_hour']} times. Escalating to a human instead of auto-contacting again."}

    action = RECOVERY_ACTION_MAP.get(txn["error_reason"], "send_reminder")
    return {"decision": "RECOVERY_ATTEMPTED", "fraud_probability": fraud_prob, "recovery_action": action,
            "reasoning": f"{txn['error_reason'].replace('_',' ').capitalize()} \u2192 {ACTION_DISPLAY.get(action, action).lower()}."}


df = generate_data()
guard_model, feature_cols, method_cols = train_guard(df)

# ============================================================
# UI
# ============================================================

logo_col, title_col = st.columns([1, 9])
with logo_col:
    st.markdown("<p style='font-size:28px;margin:0;'>\U0001F6E1\uFE0F</p>", unsafe_allow_html=True)
with title_col:
    st.markdown("<p style='font-size:22px;font-weight:600;margin:6px 0 0;'>RupeeGuard</p>", unsafe_allow_html=True)

st.markdown("#### Check any payment in seconds")
st.caption("Enter what happened. We'll tell you if it's fraud, or how we'd get the money back.")
st.write("")

col1, col2 = st.columns(2)
with col1:
    amount_rs = st.number_input("Amount (\u20b9)", min_value=1, value=1499)
    method = st.selectbox("Payment method", ["upi", "card", "netbanking", "wallet", "emi"])
    international = st.checkbox("International transaction?")
with col2:
    attempts = st.slider("Attempts in the last hour", 1, 25, 1)
    device_count = st.slider("Customers sharing this device", 1, 30, 1)
    error_reason = st.selectbox("Why did it fail?", list(RECOVERY_ACTION_MAP.keys()),
                                 format_func=lambda x: x.replace("_", " ").capitalize())

check_clicked = st.button("Check this payment", type="primary", use_container_width=True)

if check_clicked:
    txn = {
        "amount": int(amount_rs * 100), "method": method, "international": international,
        "attempts_last_hour": attempts, "device_customer_count": device_count, "error_reason": error_reason,
    }
    with st.spinner("Checking transaction..."):
        time.sleep(1.1)
        result = process_transaction(txn, guard_model, feature_cols, method_cols)

    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**\u20b9{amount_rs:,.0f}**")
            st.caption(f"{method.upper()} \u00b7 {error_reason.replace('_',' ')}")
        with c2:
            if result["decision"] == "BLOCKED":
                st.badge("Blocked", color="red")
            elif result["decision"] == "ESCALATED":
                st.badge("Sent to a human", color="orange")
            else:
                st.badge("Recovered", color="green")

        st.write("")
        st.markdown("**Fraud check**")
        prob = result["fraud_probability"]
        st.progress(min(max(prob, 0.01), 1.0))
        verdict = "blocked" if result["decision"] == "BLOCKED" else "cleared"
        st.caption(f"{prob*100:.1f}% fraud probability \u2014 {verdict}")

        if result["decision"] != "BLOCKED":
            st.write("")
            st.markdown("**Recovery action**")
            st.caption(result["reasoning"])
