"""Open data loaders used in place of Yahoo Finance.

Sources (all free, no API key):
  FRED, Federal Reserve Bank of St. Louis: https://fred.stlouisfed.org
  Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  GitHub open datasets (datasets org): https://github.com/datasets

Every download is cached as CSV under data/open_cache so runs are reproducible
and the network is only hit once.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

CACHE = Path(os.environ.get("OPENDATA_CACHE", Path(__file__).resolve().parents[1] / "data" / "open_cache"))

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
GITHUB_DATASETS = {
    "vix_daily": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
    "sp500_monthly_shiller": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "sp500_constituents": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
}


def _get(url: str, tries: int = 5) -> bytes:
    import time
    last = None
    for k in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"})
            with urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # transient network errors
            last = e
            time.sleep(2 * (k + 1))
    raise last


def _cached(key: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{key}.csv"


def fred(series: str | list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily (or native frequency) FRED series as a DataFrame indexed by date."""
    ids = [series] if isinstance(series, str) else list(series)
    cols = []
    for sid in ids:
        p = _cached(f"fred_{sid}")
        if not p.exists():
            p.write_bytes(_get(FRED_URL.format(sid=sid)))
        s = pd.read_csv(p, index_col=0, parse_dates=True, na_values=".")
        s.columns = [sid]
        cols.append(pd.to_numeric(s[sid], errors="coerce"))
    df = pd.concat(cols, axis=1).sort_index()
    df.index.name = "Date"
    return df.loc[start:end]


def french(name: str, table: int = 0, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Ken French library table as decimal returns (percent / 100).

    name: file stem such as F-F_Research_Data_Factors_daily or 10_Industry_Portfolios_daily.
    table: which block in the CSV (0 = first, value weighted for portfolio files).
    """
    p = _cached(f"french_{name}_{table}")
    if not p.exists():
        raw = _get(FRENCH_URL.format(name=name))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        blocks, cur, header = [], [], None
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) > 1 and parts[0] == "" and header is None:
                header = parts
                continue
            if header is not None and parts and parts[0].isdigit() and len(parts[0]) in (6, 8):
                cur.append(parts)
            elif header is not None and cur:
                blocks.append((header, cur))
                header, cur = None, []
            elif header is not None and not parts[0]:
                continue
        if header is not None and cur:
            blocks.append((header, cur))
        hdr, rows = blocks[table]
        df = pd.DataFrame([r[1:len(hdr)] for r in rows], index=[r[0] for r in rows], columns=hdr[1:])
        df.to_csv(p)
    df = pd.read_csv(p, index_col=0, dtype={0: str})
    idx = df.index.astype(str)
    fmt = "%Y%m%d" if len(idx[0]) == 8 else "%Y%m"
    df.index = pd.to_datetime(idx, format=fmt)
    if fmt == "%Y%m":
        df.index = df.index + pd.offsets.MonthEnd(0)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.mask(df <= -99.99) / 100.0
    return df.loc[start:end]


def github_dataset(key: str) -> pd.DataFrame:
    p = _cached(f"gh_{key}")
    if not p.exists():
        p.write_bytes(_get(GITHUB_DATASETS[key]))
    return pd.read_csv(p)


def returns_to_prices(rets: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Build a total return index from simple returns (for code that expects prices)."""
    return base * (1.0 + rets.fillna(0.0)).cumprod()


def market_prices(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Broad US equity proxies: FRED NASDAQ Composite (1971+), S&P 500 and DJIA (last 10y)."""
    return fred(["NASDAQCOM", "SP500", "DJIA"], start, end)


# ---------------------------------------------------------------------------
# Bond returns from constant maturity yields (FRED H.15)
# ---------------------------------------------------------------------------
def par_bond_returns(yields_pct: pd.Series, maturity: float) -> pd.Series:
    """Daily total return of a constant maturity par bond implied by a yield series.

    r_t = y_{t-1}/252 - D * dy + 0.5 * C * dy^2, with D and C the modified
    duration and convexity of a semiannual par bond at y_{t-1}. Standard
    approximation used to turn H.15 yields into bond index returns.
    """
    y = yields_pct.dropna() / 100.0
    dy = y.diff()
    yp = y.shift(1)
    h = yp / 2.0
    n = 2 * maturity
    # modified duration of a par bond
    D = (1 - (1 + h) ** (-n)) / yp
    C = (2 / yp**2) * (1 - (1 + h) ** (-n)) - (2 * maturity / (yp * (1 + h) ** (n + 1)))
    r = yp / 252.0 - D * dy + 0.5 * C * dy**2
    return r.dropna()


# ---------------------------------------------------------------------------
# Open data stand ins for common ETFs and tickers
# ---------------------------------------------------------------------------
PROXY_DESCRIPTIONS = {
    "SPY": "US total market, Ken French Mkt-RF + RF (daily, value weighted)",
    "VTI": "US total market, Ken French Mkt-RF + RF",
    "QQQ": "NASDAQ-100 index, FRED NASDAQ100 (price index)",
    "IWM": "US small caps, Ken French bottom 30% size portfolio (value weighted)",
    "EFA": "Developed ex US market, Ken French Developed ex US Mkt-RF + RF (USD)",
    "DIA": "Dow Jones Industrial Average, FRED DJIA (price index, 10y history)",
    "XLF": "Ken French 12 industry: Money (finance)",
    "XLE": "Ken French 12 industry: Enrgy",
    "XLK": "Ken French 12 industry: BusEq (business equipment, tech)",
    "XLV": "Ken French 12 industry: Hlth",
    "XLI": "Ken French 12 industry: Manuf",
    "XLP": "Ken French 12 industry: NoDur (consumer staples)",
    "XLY": "Ken French 12 industry: Shops (retail, consumer discretionary)",
    "XLU": "Ken French 12 industry: Utils",
    "XLB": "Ken French 12 industry: Chems (materials)",
    "XLC": "Ken French 12 industry: Telcm",
    "SHY": "2y constant maturity Treasury par bond from FRED DGS2",
    "IEF": "10y constant maturity Treasury par bond from FRED DGS10",
    "TLT": "20y constant maturity Treasury par bond from FRED DGS20",
    "LQD": "Baa corporate par bond (12y) from FRED DBAA (Moody's seasoned Baa yield)",
    "USO": "WTI crude spot, FRED DCOILWTICO",
    "BTC-USD": "Bitcoin, FRED CBBTCUSD (Coinbase)",
    "^VIX": "CBOE VIX close, FRED VIXCLS",
    "^GSPC": "S&P 500 index, FRED SP500 (price index, 10y history)",
    "^IXIC": "NASDAQ Composite, FRED NASDAQCOM (price index, 1971+)",
}

_IND12 = {"XLF": "Money", "XLE": "Enrgy", "XLK": "BusEq", "XLV": "Hlth", "XLI": "Manuf",
          "XLP": "NoDur", "XLY": "Shops", "XLU": "Utils", "XLB": "Chems", "XLC": "Telcm"}


def proxy_returns(tickers: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily simple returns for open data stand ins of the given tickers.

    Tickers without an open daily source are skipped (a warning lists them).
    """
    out = {}
    missing = []
    ff = None
    for t in tickers:
        if t in ("SPY", "VTI"):
            ff = ff if ff is not None else french("F-F_Research_Data_Factors_daily")
            out[t] = ff["Mkt-RF"] + ff["RF"]
        elif t == "IWM":
            out[t] = french("Portfolios_Formed_on_ME_daily")["Lo 30"]
        elif t == "EFA":
            d = french("Developed_ex_US_3_Factors_Daily")
            out[t] = d["Mkt-RF"] + d["RF"]
        elif t in _IND12:
            out[t] = french("12_Industry_Portfolios_daily")[_IND12[t]]
        elif t == "QQQ":
            out[t] = fred("NASDAQ100")["NASDAQ100"].dropna().pct_change()
        elif t == "DIA":
            out[t] = fred("DJIA")["DJIA"].dropna().pct_change()
        elif t == "^GSPC":
            out[t] = fred("SP500")["SP500"].dropna().pct_change()
        elif t == "^IXIC":
            out[t] = fred("NASDAQCOM")["NASDAQCOM"].dropna().pct_change()
        elif t == "^VIX":
            out[t] = fred("VIXCLS")["VIXCLS"].dropna().pct_change()
        elif t == "SHY":
            out[t] = par_bond_returns(fred("DGS2")["DGS2"], 2)
        elif t == "IEF":
            out[t] = par_bond_returns(fred("DGS10")["DGS10"], 10)
        elif t == "TLT":
            out[t] = par_bond_returns(fred("DGS20")["DGS20"], 20)
        elif t == "LQD":
            out[t] = par_bond_returns(fred("DBAA")["DBAA"], 12)
        elif t == "USO":
            s = fred("DCOILWTICO")["DCOILWTICO"].dropna()
            out[t] = s[s > 0].pct_change()
        elif t == "BTC-USD":
            out[t] = fred("CBBTCUSD")["CBBTCUSD"].dropna().pct_change()
        else:
            missing.append(t)
    if missing:
        import warnings
        warnings.warn(f"No open daily source for {missing}; dropped from universe")
    df = pd.DataFrame(out).sort_index()
    df = df.loc[start:end]
    # keep only dates where the US equity calendar has data for most assets
    df = df.dropna(thresh=max(1, int(0.6 * df.shape[1])))
    return df


def proxy_prices(tickers: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Total return index (base 100) built from proxy_returns, for price based code."""
    r = proxy_returns(tickers, start, end)
    p = 100.0 * (1.0 + r.fillna(0.0)).cumprod()
    started = r.notna().cummax()
    return p.where(started)


# ---------------------------------------------------------------------------
# Data snapshot manifest (reproducibility)
# ---------------------------------------------------------------------------
def write_manifest(path: str | os.PathLike) -> dict:
    """Record every cached input file with its SHA-256, size, rows and date range.

    Ken French and FRED revise history from time to time. Committing this
    manifest next to the results shows exactly which data they came from, and
    verify_manifest() tells you whether a fresh download still matches.
    """
    import datetime as _dt
    import hashlib
    import json

    files = {}
    for p in sorted(CACHE.glob("*.csv")):
        raw = p.read_bytes()
        info = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        try:
            df = pd.read_csv(p, index_col=0)
            info["rows"] = int(len(df))
            info["first"] = str(df.index[0])
            info["last"] = str(df.index[-1])
        except Exception:  # pragma: no cover - unusual file
            pass
        files[p.name] = info
    out = {"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sources": {"fred": FRED_URL, "french": FRENCH_URL, "github": GITHUB_DATASETS},
           "files": files}
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    return out


def verify_manifest(path: str | os.PathLike) -> dict:
    """Compare cached files with a manifest. Returns {file: 'ok' | 'changed' | 'missing'}."""
    import hashlib
    import json

    m = json.loads(Path(path).read_text())
    status = {}
    for name, info in m["files"].items():
        p = CACHE / name
        if not p.exists():
            status[name] = "missing"
        else:
            status[name] = "ok" if hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"] else "changed"
    return status
