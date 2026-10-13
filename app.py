"""Latest US Treasury yield curve, fitted from FRED par yields."""

import os

import matplotlib

matplotlib.use("Agg")

import streamlit as st

from curves import YieldCurve, load_treasury

st.set_page_config(page_title="US Treasury yield curve", layout="wide")

API_KEY = '424034afb86bdcab88017369d65d00da'

def fred_api_key():
    return API_KEY
    


@st.cache_data(ttl=12 * 60 * 60, show_spinner="Loading Treasury par yields from FRED...")
def cached_treasury(api_key):
    return load_treasury(api_key)


api_key = fred_api_key()
if not api_key:
    st.error("Set FRED_API_KEY in the environment or in .streamlit/secrets.toml.")
    st.stop()

treasury = cached_treasury(api_key)
# Newest first, so the default is the latest FRED date.
as_of = st.sidebar.selectbox(
    "As-of date",
    options=treasury.index[::-1],
    index=0,
    format_func=lambda ts: ts.strftime("%Y-%m-%d"),
)
method = st.sidebar.selectbox(
    "Fitting method",
    options=list(YieldCurve.METHODS),
    index=0,
    format_func=lambda name: YieldCurve.METHODS[name],
)

curve = YieldCurve(treasury.loc[as_of], method)
st.title("US Treasury yield curve")
st.caption(f"{curve.name} · {as_of.strftime('%Y-%m-%d')}")

if curve.is_arbitrage_free():
    st.success("Arbitrage free: the instantaneous forward rate stays positive.")
else:
    st.warning("Not arbitrage free: the instantaneous forward rate goes negative.")

st.pyplot(curve.figure())

quotes = curve.quotes.rename(
    columns={
        "par_rate_pct": "Par yield (%)",
        "maturity_years": "Maturity (years)",
        "yield_cc_pct": "Continuous zero (%)",
        "discount_factor": "Discount factor",
    }
)
st.dataframe(quotes)
