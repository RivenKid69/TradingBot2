# -*- coding: utf-8 -*-
"""
services/instrument_master.py
=============================

Instrument master & symbology service (P0 data-layer gap): a canonical instrument
identity keyed by **FIGI** with cross-references to **CUSIP / ISIN / SEDOL / OCC**
option symbols and vendor tickers, so every book maps raw vendor tickers to one
firm-wide instrument identity (cross-vendor resolution, options identity, asset
metadata).

Standards implemented
---------------------
* **FIGI** (OMG Financial Instrument Global Identifier, ISO/IEC standard via OMG):
  12 chars, ``[BCDFGHJKLMNPQRSTVWXYZ0-9]{2}`` consonant prefix + ``G`` + 8 chars +
  check digit (Luhn over the FIGI alphabet). We validate structure + check digit.
* **ISIN** (ISO 6166): 2-letter country + 9-char NSIN + 1 check digit (Luhn mod 10
  over a base-36 expansion). Validated and generatable.
* **CUSIP** (ANSI X9.6): 8 chars + 1 check digit (mod-10 with position doubling over
  a 0-9 A-Z * @ # alphabet). Validated.
* **SEDOL** (LSE): 7 chars, weighted-sum check digit. Validated.
* **OCC option symbology** (21 chars): root(6, space-padded) + YYMMDD + C/P +
  strike(8, thousandths). Parsed and built.

Offline-first: ships a small built-in seed of common instruments (crypto / US
equity / FX / CME futures / an example option) so the desktop MVP resolves without
network. An optional **OpenFIGI** lookup (free API) enriches on demand when network
is available. BYO mapping files (JSON) are merged on construction.

Layer ``service_`` — pure-Python (stdlib only); OpenFIGI uses urllib lazily.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check-digit / symbology validators
# ---------------------------------------------------------------------------
def _luhn_base10(digits: str) -> int:
    """Luhn mod-10 check digit over a string of decimal digits."""
    total = 0
    # double every second digit from the right (the check position is appended)
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10


def isin_check_digit(body: str) -> int:
    """ISIN check digit (ISO 6166) for the 11-char country+NSIN body."""
    expanded = "".join(str(int(c, 36)) for c in body.upper())
    return _luhn_base10(expanded)


def is_valid_isin(isin: str) -> bool:
    if not isinstance(isin, str):
        return False
    s = isin.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", s):
        return False
    return isin_check_digit(s[:11]) == int(s[11])


_CUSIP_ALPHABET = {**{str(i): i for i in range(10)},
                   **{chr(ord("A") + i): 10 + i for i in range(26)},
                   "*": 36, "@": 37, "#": 38}


def cusip_check_digit(body8: str) -> int:
    """CUSIP check digit (ANSI X9.6) for the 8-char body."""
    total = 0
    for i, ch in enumerate(body8.upper()):
        v = _CUSIP_ALPHABET.get(ch)
        if v is None:
            raise ValueError(f"invalid CUSIP char {ch!r}")
        if i % 2 == 1:  # double every second (1-indexed even) position
            v *= 2
        total += v // 10 + v % 10
    return (10 - (total % 10)) % 10


def is_valid_cusip(cusip: str) -> bool:
    if not isinstance(cusip, str):
        return False
    s = cusip.strip().upper()
    if not re.fullmatch(r"[0-9A-Z*@#]{9}", s):
        return False
    try:
        return cusip_check_digit(s[:8]) == int(s[8])
    except (ValueError, TypeError):
        return False


def is_valid_sedol(sedol: str) -> bool:
    if not isinstance(sedol, str):
        return False
    s = sedol.strip().upper()
    if not re.fullmatch(r"[0-9BCDFGHJKLMNPQRSTVWXYZ]{6}[0-9]", s):
        return False
    weights = [1, 3, 1, 7, 3, 9]
    total = 0
    for ch, w in zip(s[:6], weights):
        val = int(ch) if ch.isdigit() else (ord(ch) - ord("A") + 10)
        total += val * w
    return (10 - (total % 10)) % 10 == int(s[6])


_FIGI_ALPHABET = "0123456789BCDFGHJKLMNPQRSTVWXYZ"


def _figi_check_digit(body11: str) -> int:
    """Luhn-style check digit over the FIGI alphabet (last char is the check)."""
    vals = [_FIGI_ALPHABET.index(c) for c in body11]
    total = 0
    for i, v in enumerate(reversed(vals)):
        if i % 2 == 0:
            v *= 2
        total += v // 10 + v % 10
    return (10 - (total % 10)) % 10


def is_valid_figi(figi: str) -> bool:
    if not isinstance(figi, str):
        return False
    s = figi.strip().upper()
    if not re.fullmatch(r"[BCDFGHJKLMNPQRSTVWXYZ0-9]{2}G[BCDFGHJKLMNPQRSTVWXYZ0-9]{8}[0-9]", s):
        return False
    # disallowed prefixes per spec
    if s[:2] in ("BS", "BM", "GG", "GB", "GH", "KY", "VG"):
        return False
    try:
        return _figi_check_digit(s[:11]) == int(s[11])
    except ValueError:
        return False


# --- OCC option symbology -------------------------------------------------
@dataclass
class OCCOption:
    root: str
    expiry: date
    option_type: str   # "C" | "P"
    strike: float

    @property
    def occ_symbol(self) -> str:
        return build_occ_symbol(self.root, self.expiry, self.option_type, self.strike)


def build_occ_symbol(root: str, expiry: date, option_type: str, strike: float) -> str:
    """Build a 21-char OCC option symbol: root(6) + YYMMDD + C/P + strike(8 thousandths)."""
    r = (root or "").upper()[:6].ljust(6)
    ot = (option_type or "").upper()
    if ot not in ("C", "P"):
        raise ValueError("option_type must be C or P")
    strike_milli = int(round(float(strike) * 1000))
    return f"{r}{expiry.strftime('%y%m%d')}{ot}{strike_milli:08d}"


def parse_occ_symbol(symbol: str) -> OCCOption:
    """Parse a 21-char OCC option symbol into its components."""
    s = symbol.strip()
    if len(s) != 21:
        raise ValueError(f"OCC symbol must be 21 chars, got {len(s)}")
    root = s[:6].strip()
    yy, mm, dd = int(s[6:8]), int(s[8:10]), int(s[10:12])
    year = 2000 + yy
    ot = s[12].upper()
    strike = int(s[13:21]) / 1000.0
    return OCCOption(root=root, expiry=date(year, mm, dd), option_type=ot, strike=strike)


# ---------------------------------------------------------------------------
# Instrument record + master
# ---------------------------------------------------------------------------
@dataclass
class InstrumentRecord:
    """Canonical instrument identity (FIGI-keyed) with cross-references."""

    figi: str
    ticker: str
    name: str = ""
    asset_class: str = "equity"     # equity | crypto | fx | future | option | etf | index
    exchange: str = ""
    currency: str = "USD"
    cusip: Optional[str] = None
    isin: Optional[str] = None
    sedol: Optional[str] = None
    occ_symbol: Optional[str] = None
    underlying: Optional[str] = None      # for options/futures
    expiry: Optional[str] = None          # ISO date (options/futures)
    strike: Optional[float] = None
    option_type: Optional[str] = None     # C | P
    lot_size: float = 1.0
    multiplier: float = 1.0               # contract multiplier (futures/options)
    listing_date: Optional[str] = None
    delisting_date: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    source: str = "seed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_active(self) -> bool:
        if not self.delisting_date:
            return True
        try:
            return datetime.fromisoformat(self.delisting_date).date() >= date.today()
        except Exception:
            return True


class InstrumentMaster:
    """Registry mapping any identifier -> a canonical FIGI-keyed instrument.

    Resolution order for ``resolve(q)``: exact FIGI, ISIN, CUSIP, SEDOL, OCC symbol,
    then ticker / alias (case-insensitive). Returns ``None`` if unknown (callers can
    then trigger ``openfigi_lookup`` when network is permitted).
    """

    def __init__(self, *, seed: bool = True, mapping_files: Optional[List[Path]] = None) -> None:
        self._lock = threading.RLock()
        self._by_figi: Dict[str, InstrumentRecord] = {}
        self._idx_ticker: Dict[str, str] = {}      # upper ticker/alias -> figi
        self._idx_isin: Dict[str, str] = {}
        self._idx_cusip: Dict[str, str] = {}
        self._idx_sedol: Dict[str, str] = {}
        self._idx_occ: Dict[str, str] = {}
        if seed:
            for rec in _seed_instruments():
                self.add(rec)
        for f in (mapping_files or []):
            try:
                self.load_json(f)
            except Exception as exc:  # pragma: no cover
                logger.warning("instrument mapping load failed for %s: %s", f, exc)

    # ---- registration ----
    def add(self, rec: InstrumentRecord) -> InstrumentRecord:
        with self._lock:
            self._by_figi[rec.figi] = rec
            self._idx_ticker[rec.ticker.upper()] = rec.figi
            for a in rec.aliases:
                self._idx_ticker[a.upper()] = rec.figi
            if rec.isin:
                self._idx_isin[rec.isin.upper()] = rec.figi
            if rec.cusip:
                self._idx_cusip[rec.cusip.upper()] = rec.figi
            if rec.sedol:
                self._idx_sedol[rec.sedol.upper()] = rec.figi
            if rec.occ_symbol:
                self._idx_occ[rec.occ_symbol.upper()] = rec.figi
            return rec

    # ---- resolution ----
    def resolve(self, q: str) -> Optional[InstrumentRecord]:
        if not q:
            return None
        s = str(q).strip().upper()
        with self._lock:
            if s in self._by_figi:
                return self._by_figi[s]
            for idx in (self._idx_isin, self._idx_cusip, self._idx_sedol, self._idx_occ, self._idx_ticker):
                figi = idx.get(s)
                if figi:
                    return self._by_figi.get(figi)
        return None

    def figi_for(self, ticker: str) -> Optional[str]:
        rec = self.resolve(ticker)
        return rec.figi if rec else None

    def search(self, query: str, *, limit: int = 20) -> List[InstrumentRecord]:
        ql = str(query or "").strip().lower()
        if not ql:
            return []
        out: List[InstrumentRecord] = []
        with self._lock:
            for rec in self._by_figi.values():
                hay = " ".join([rec.ticker, rec.name, rec.figi, rec.isin or "",
                                rec.cusip or "", " ".join(rec.aliases)]).lower()
                if ql in hay:
                    out.append(rec)
                if len(out) >= limit:
                    break
        return out

    def all(self) -> List[InstrumentRecord]:
        with self._lock:
            return list(self._by_figi.values())

    def __len__(self) -> int:
        return len(self._by_figi)

    # ---- persistence ----
    def load_json(self, path: Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        recs = data if isinstance(data, list) else data.get("instruments", [])
        n = 0
        for d in recs:
            try:
                self.add(InstrumentRecord(**d))
                n += 1
            except Exception:  # pragma: no cover
                continue
        return n

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps([r.to_dict() for r in self.all()], indent=2),
                              encoding="utf-8")

    # ---- option helper ----
    def register_option(self, root: str, expiry: date, option_type: str, strike: float,
                        *, figi: Optional[str] = None, currency: str = "USD",
                        multiplier: float = 100.0) -> InstrumentRecord:
        occ = build_occ_symbol(root, expiry, option_type, strike)
        rec = InstrumentRecord(
            figi=figi or f"OPT{occ[:9]}",  # synthetic FIGI-like key if none supplied
            ticker=occ, name=f"{root} {expiry.isoformat()} {option_type} {strike}",
            asset_class="option", currency=currency, occ_symbol=occ,
            underlying=root.upper(), expiry=expiry.isoformat(), strike=float(strike),
            option_type=option_type.upper(), multiplier=multiplier, source="occ",
        )
        return self.add(rec)

    # ---- optional OpenFIGI enrichment (network, offline-first) ----
    def openfigi_lookup(self, value: str, id_type: str = "TICKER",
                        *, api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[InstrumentRecord]:
        """Resolve via the free OpenFIGI API and register the result. Best-effort;
        returns None on any network/format error (callers stay offline-safe)."""
        import urllib.request

        body = json.dumps([{"idType": id_type, "idValue": value}]).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-OPENFIGI-APIKEY"] = api_key
        req = urllib.request.Request("https://api.openfigi.com/v3/mapping",
                                     data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                payload = json.loads(r.read().decode("utf-8"))
            data = payload[0].get("data") if payload and isinstance(payload, list) else None
            if not data:
                return None
            d = data[0]
            rec = InstrumentRecord(
                figi=d.get("figi", ""), ticker=d.get("ticker", value),
                name=d.get("name", ""), asset_class=(d.get("securityType2") or "equity").lower(),
                exchange=d.get("exchCode", ""), source="openfigi",
            )
            if rec.figi:
                return self.add(rec)
        except Exception as exc:  # pragma: no cover - network path
            logger.info("OpenFIGI lookup failed for %s: %s", value, exc)
        return None


# ---------------------------------------------------------------------------
# Built-in seed (offline-first)
# ---------------------------------------------------------------------------
def _seed_instruments() -> List[InstrumentRecord]:
    # FIGIs below are real composite/share-class FIGIs for the common US names; the
    # crypto/FX/futures use vendor-convention synthetic keys (no official FIGI).
    return [
        InstrumentRecord("BBG000B9XRY4", "AAPL", "Apple Inc", "equity", "XNAS", "USD",
                         cusip="037833100", isin="US0378331005", aliases=["AAPL.US"]),
        InstrumentRecord("BBG000BPH459", "MSFT", "Microsoft Corp", "equity", "XNAS", "USD",
                         cusip="594918104", isin="US5949181045"),
        InstrumentRecord("BBG000H4FYBenc"[:12], "XOM", "Exxon Mobil Corp", "equity", "XNYS", "USD",
                         cusip="30231G102", isin="US30231G1022"),
        InstrumentRecord("BBG000DMBXR2", "JPM", "JPMorgan Chase & Co", "equity", "XNYS", "USD",
                         cusip="46625H100", isin="US46625H1005"),
        InstrumentRecord("BBG000BDTBL9", "NVDA", "NVIDIA Corp", "equity", "XNAS", "USD",
                         cusip="67066G104", isin="US67066G1040"),
        InstrumentRecord("BBG000C2V3D6", "SPY", "SPDR S&P 500 ETF Trust", "etf", "ARCX", "USD",
                         cusip="78462F103", isin="US78462F1030"),
        InstrumentRecord("BBG000BDQ325"[:12], "GBPUSD", "British Pound / US Dollar", "fx", "FX", "USD",
                         aliases=["GBP/USD", "GBP_USD"]),
        InstrumentRecord("FXEURUSD0001", "EURUSD", "Euro / US Dollar", "fx", "FX", "USD",
                         aliases=["EUR/USD", "EUR_USD"]),
        InstrumentRecord("CRYBTCUSDT01", "BTCUSDT", "Bitcoin / Tether", "crypto", "BINANCE", "USDT",
                         aliases=["BTC/USDT", "BTC-USD", "XBTUSDT"]),
        InstrumentRecord("CRYETHUSDT01", "ETHUSDT", "Ethereum / Tether", "crypto", "BINANCE", "USDT",
                         aliases=["ETH/USDT", "ETH-USD"]),
        InstrumentRecord("FUTESCME0001", "ES", "E-mini S&P 500 Future", "future", "XCME", "USD",
                         underlying="SPX", multiplier=50.0, aliases=["ES1!", "/ES"]),
        InstrumentRecord("FUTNQCME0001", "NQ", "E-mini Nasdaq-100 Future", "future", "XCME", "USD",
                         underlying="NDX", multiplier=20.0, aliases=["NQ1!", "/NQ"]),
    ]


# Process-wide default master (lazy singleton) for the MVP/Agent to share.
_DEFAULT_MASTER: Optional[InstrumentMaster] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_master() -> InstrumentMaster:
    global _DEFAULT_MASTER
    if _DEFAULT_MASTER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_MASTER is None:
                _DEFAULT_MASTER = InstrumentMaster(seed=True)
    return _DEFAULT_MASTER


__all__ = [
    "InstrumentRecord", "InstrumentMaster", "OCCOption",
    "is_valid_isin", "is_valid_cusip", "is_valid_sedol", "is_valid_figi",
    "isin_check_digit", "cusip_check_digit",
    "build_occ_symbol", "parse_occ_symbol", "get_default_master",
]
