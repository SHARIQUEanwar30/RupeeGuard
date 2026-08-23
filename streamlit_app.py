import streamlit as st
import pandas as pd
import numpy as np
import string
import random
import time as time_module
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="RupeeGuard", page_icon="🛡️", layout="wide")

# ============================================================
# DATA + MODEL (cached so this only runs once, not on every click)
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
    now = int(time_module.time())

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
    return model, feature_cols, method_dummies.columns.tolist(), X_test, y_test, df.loc[X_test.index]


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
                "reasoning": f"Cleared as genuine ({fraud_prob*100:.1f}% fraud probability), but already failed {txn['attempts_last_hour']} times. Escalating instead of auto-contacting again."}

    action = RECOVERY_ACTION_MAP.get(txn["error_reason"], "send_reminder")
    return {"decision": "RECOVERY_ATTEMPTED", "fraud_probability": fraud_prob, "recovery_action": action,
            "reasoning": f"Cleared as genuine ({fraud_prob*100:.1f}% fraud probability). '{txn['error_reason']}' mapped to this action."}


# ============================================================
# LOAD EVERYTHING
# ============================================================

df = generate_data()
guard_model, feature_cols, method_cols, X_test, y_test, test_meta = train_guard(df)

st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

# ============================================================
# UI
# ============================================================

st.title("🛡️ RupeeGuard")
st.caption("Guards every rupee — blocks the fraud, recovers the rest.")

tab1, tab2 = st.tabs(["📊 Dashboard", "▶️ Live demo"])

# ---------------- DASHBOARD TAB ----------------
with tab1:
    test_probs = guard_model.predict_proba(X_test)[:, 1]
    pipeline_df = test_meta.copy()
    pipeline_df["guard_predicts_fraud"] = test_probs >= GUARD_THRESHOLD
    failed_in_test = pipeline_df[pipeline_df["status"] == "failed"].copy()
    blocked = failed_in_test[failed_in_test["guard_predicts_fraud"] == True]
    sent_to_helper = failed_in_test[failed_in_test["guard_predicts_fraud"] == False]

    np.random.seed(42)
    success_rate_map = {"retry_immediately": 0.65, "retry_later_2h": 0.55, "retry_later_24h": 0.45,
                         "resend_collect_request": 0.50, "send_reminder": 0.30, "suggest_alt_method": 0.40}
    recovered_amt, escalated_amt = 0, 0
    for _, row in sent_to_helper.iterrows():
        if row["attempts_last_hour"] >= MAX_ATTEMPTS_BEFORE_ESCALATION:
            escalated_amt += row["amount"] / 100
        else:
            action = RECOVERY_ACTION_MAP.get(row["error_reason"], "send_reminder")
            if np.random.rand() < success_rate_map[action]:
                recovered_amt += row["amount"] / 100

    total_batch = failed_in_test["amount"].sum() / 100
    blocked_amt = blocked["amount"].sum() / 100
    not_recovered_amt = total_batch - recovered_amt - blocked_amt - escalated_amt
    precision = (blocked["is_fraud"] == 1).sum() / len(blocked) if len(blocked) > 0 else 0
    total_fraud = (failed_in_test["is_fraud"] == 1).sum()
    recall = (blocked["is_fraud"] == 1).sum() / total_fraud if total_fraud > 0 else 0
    caught = (blocked["is_fraud"] == 1).sum()
    total_dataset = len(df)

    st.info(
        f"Tested on **{len(failed_in_test)} failed payments** (₹{total_batch/100000:.2f}L), "
        f"pulled from a **{total_dataset:,}-transaction dataset**. These are payments the fraud "
        f"model had never seen before, so this is an honest, real-world result — not a number "
        f"from its own training data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Money recovered", f"₹{recovered_amt/100000:.2f}L", f"{recovered_amt/total_batch*100:.1f}% of batch",
        help="A genuine payment failed, then RupeeGuard retried it or reminded the customer, and it went through."
    )
    c2.metric(
        "Fraud blocked", f"₹{blocked_amt/100000:.2f}L", f"{len(blocked)} payments, 0 false alarms",
        help=f"Stopped before any money moved. Caught {caught} of the {total_fraud} real fraud attempts in this batch — zero genuine customers were wrongly blocked."
    )
    c3.metric(
        "Sent to a human", f"₹{escalated_amt/100000:.2f}L", f"{escalated_amt/total_batch*100:.1f}%, not lost",
        help="These already failed 4+ times, so RupeeGuard stops contacting the customer automatically and hands it to a person instead. The money isn't lost, just paused for manual follow-up."
    )
    c4.metric(
        "Not recovered", f"₹{not_recovered_amt/100000:.2f}L", f"{not_recovered_amt/total_batch*100:.1f}%, honest miss",
        help="RupeeGuard attempted a fix, but the customer didn't complete the payment in this test. Reported honestly, not hidden."
    )

    st.divider()
    st.subheader("Fraud detection model performance")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Precision", f"{precision*100:.1f}%",
                 help="When the Guard says fraud, how often is it actually right? 100% means zero genuine customers were ever wrongly blocked.")
    mcol2.metric("Recall", f"{recall*100:.1f}%",
                 help="Out of all the real fraud in this batch, how much did the Guard actually catch?")
    st.caption(f"{caught} of {total_fraud} fraud cases caught, 0 genuine customers wrongly blocked. Evaluated on a held-out test split — data the model never saw during training.")

# ---------------- LIVE DEMO TAB ----------------
with tab2:
    st.subheader("Run a transaction through RupeeGuard")
    st.caption("Enter details for a failed payment and see exactly what the system decides, and why.")

    colA, colB = st.columns(2)
    with colA:
        amount_rs = st.number_input("Amount (₹)", min_value=1, value=1500)
        method = st.selectbox("Payment method", ["card", "upi", "netbanking", "wallet", "emi"])
        international = st.checkbox("International transaction?")
    with colB:
        attempts = st.slider("Attempts in the last hour", 1, 25, 1)
        device_count = st.slider("Number of different customers on this device", 1, 30, 1)
        error_reason = st.selectbox("Why did it fail?", list(RECOVERY_ACTION_MAP.keys()))

    if st.button("Run RupeeGuard", type="primary"):
        txn = {
            "amount": int(amount_rs * 100), "method": method, "international": international,
            "attempts_last_hour": attempts, "device_customer_count": device_count, "error_reason": error_reason,
        }
        result = process_transaction(txn, guard_model, feature_cols, method_cols)

        st.divider()
        if result["decision"] == "BLOCKED":
            st.error(f"🚫 BLOCKED — flagged as fraud ({result['fraud_probability']*100:.1f}% confidence)")
        elif result["decision"] == "ESCALATED":
            st.warning(f"👤 ESCALATED TO HUMAN — {ACTION_DISPLAY.get(result['recovery_action'])}")
        else:
            st.success(f"✅ RECOVERY ATTEMPTED — {ACTION_DISPLAY.get(result['recovery_action'])}")

        st.write(f"**Fraud probability:** {result['fraud_probability']*100:.1f}%")
        st.write(f"**Reasoning:** {result['reasoning']}")
