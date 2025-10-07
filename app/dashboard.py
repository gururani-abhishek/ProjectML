import smtplib
from email.message import EmailMessage
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

def send_alert_email(to_email: str, itsoname: str, alerts: list):
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    from_email = os.getenv("FROM_EMAIL", smtp_user)
    to_email = os.getenv("TEST_EMAIL")
    subject = f"ALERT: SDLC Gap Risk for ITSO {itsoname}"
    body = f"Dear {itsoname},\n\nThe following application under your responsibility have a predicted SDLC gap risk:\n\n"
    for alert in alerts:
        prob_percent = float(alert['Probability']) * 100
        body += (
            f"Application ID: {alert['Application']}\n"
            f"Control: {alert['Control']}\n"
            f"Predicted Probability of Gap: {prob_percent:.1f}%\n\n"
        )
    body += "Please review the SDLC controls for your applications.\n\nRegards,\nSDLC alerts Team"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"Alert email sent to {to_email} for ITSO {itsoname}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

def natural_control_sort(names):
    def key_fn(x):
        m = re.search(r'(\d+)$', x)
        return int(m.group(1)) if m else 0
    return sorted(names, key=key_fn)

@st.cache_data
def load_predictions():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("per_control_predictions_current_")]
    if not files:
        st.error("No prediction file found. Run main.py first.")
        return None
    latest = sorted(files)[-1]
    return pd.read_csv(os.path.join(OUTPUT_DIR, latest), parse_dates=["feature_month","predicts_for_month"])

@st.cache_data
def load_feature_month_snapshot(feature_month_str: str):
    path = os.path.join(DATA_DIR, f"SDLC_{feature_month_str}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)[["applicationID","ITSO_Name"]]

def reshape_long(df):
    prob_cols = [c for c in df.columns if c.startswith("pred_next_gap_prob_SDLC")]
    long = df.melt(
        id_vars=["applicationID","feature_month","predicts_for_month"],
        value_vars=prob_cols,
        var_name="control",
        value_name="prob"
    )
    long["control"] = long["control"].str.replace("pred_next_gap_prob_", "")
    return long

def alert_send(alerts_df):
    if alerts_df.empty:
        st.warning("No alerts to send.")
        return
    grouped = alerts_df.groupby("ITSO")
    sent_count = 0
    for itsoname, group in grouped:
        alerts_list = group[["Application","Control","Probability"]].to_dict("records")
        to_email = f"{itsoname.replace(' ','.').lower()}@example.com"
        send_alert_email(to_email, itsoname, alerts_list)
        sent_count += 1
    st.info(f"Alert emails sent to {sent_count} ITSO(s).")

def main():
    st.title("SDLC Control Gap Predictions Dashboard")
    df = load_predictions()
    if df is None:
        return

    feature_month_str = df["feature_month"].dt.strftime("%Y%m").iloc[0]
    if "ITSO_Name" not in df.columns:
        snap = load_feature_month_snapshot(feature_month_str)
        if snap is not None:
            df = df.merge(snap, on="applicationID", how="left")
        else:
            df["ITSO_Name"] = ""

    st.caption(f"Feature month: {df['feature_month'].iloc[0].date()} | Predicting: {df['predicts_for_month'].iloc[0].date()}")

    prob_cols = [c for c in df.columns if c.startswith("pred_next_gap_prob_SDLC")]
    st.subheader("Applications Overview")
    st.metric("Apps", len(df))

    control_list = natural_control_sort([c.replace("pred_next_gap_prob_","") for c in prob_cols])
    raw_choice = st.multiselect("Select controls :", ["ALL"] + control_list, default=["SDLC1","SDLC2"])
    control_choice = control_list if "ALL" in raw_choice else raw_choice

    threshold = st.slider("Alert threshold", 0.0, 1.0, 0.6, 0.01)
    sel_cols = [f"pred_next_gap_prob_{c}" for c in control_choice]
    if not sel_cols:
        st.warning("Select at least one control.")
        return

    df["selected_max"] = df[sel_cols].max(axis=1)

    st.subheader("Top Risky Apps")
    top_n = st.number_input("Top N apps by selected controls' max prob", 1, len(df), min(15, len(df)), 1)
    top_table = df.sort_values("selected_max", ascending=False).head(top_n)

    display_cols = ["applicationID", "ITSO_Name", "selected_max"] + sel_cols
    show = top_table[display_cols].copy()
    show["selected_max"] = show["selected_max"].map(lambda x: f"{x:.3f}")
    for c in sel_cols:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    rename_map = {c: c.replace("pred_next_gap_prob_","") for c in sel_cols}
    show.rename(columns=rename_map, inplace=True)
    st.dataframe(show.reset_index(drop=True), hide_index=True, use_container_width=True)

    st.subheader("Heatmap (Apps vs Controls)")
    long = reshape_long(df)
    long_disp = long[long["control"].isin(control_choice)] if control_choice else long
    heat_pivot = long_disp.pivot(index="applicationID", columns="control", values="prob")
    fig = px.imshow(
        heat_pivot,
        aspect="auto",
        color_continuous_scale="Reds",
        origin="lower",
        title="Predicted Next-Month Gap Probability"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Alerts Above Threshold")
    alerts = long[(long["prob"] >= threshold)].copy()
    alerts = alerts.merge(df[["applicationID","ITSO_Name"]].drop_duplicates(), on="applicationID", how="left")
    alerts = alerts.sort_values("prob", ascending=False).head(50)
    alerts["prob"] = alerts["prob"].map(lambda x: f"{x:.3f}")
    alerts.rename(columns={
        "applicationID": "Application",
        "ITSO_Name": "ITSO",
        "control": "Control",
        "prob": "Probability",
        "feature_month": "Feature_Month",
        "predicts_for_month": "Predicts_For_Month"
    }, inplace=True)
    display_cols = ["Application","ITSO","Control","Probability","Feature_Month","Predicts_For_Month"]
    if st.button("Send Alerts"):
        alert_send(alerts)
    st.write(f"Controls above {threshold:.2f}: {len(alerts)} rows")
    st.dataframe(alerts[display_cols].reset_index(drop=True), hide_index=True, use_container_width=True)

    st.download_button(
        "Download Current Predictions (CSV)",
        data=df.to_csv(index=False),
        file_name="current_per_control_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
