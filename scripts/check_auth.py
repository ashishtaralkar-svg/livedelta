"""Read-only AUTHENTICATED check — finds WHICH Delta platform a key belongs to.

Makes a SIGNED /v2/positions/margined call (no orders) against BOTH the Global and
India production endpoints with the key/secret from .env, and reports the result
for each. This pinpoints where the key was actually created:

  GLOBAL=not found, INDIA=found/ip-blocked  -> it's an INDIA key
  GLOBAL=SUCCESS                            -> valid Global key
  both=not found                            -> key not found anywhere (typo/activation)

    python scripts/check_auth.py
"""

from __future__ import annotations

from deltabot.config import load_settings
from deltabot.exchange.rest_client import DeltaRestError, RestClient
from deltabot.logging_setup import setup_logging

ENDPOINTS = [
    ("GLOBAL", "https://api.delta.exchange"),
    ("INDIA", "https://api.india.delta.exchange"),
]


def main() -> None:
    s = load_settings()
    setup_logging("WARNING")
    key = s.api_key.get_secret_value()
    secret = s.api_secret.get_secret_value()
    print(f"Testing key {key[:4]}...{key[-4:]} (len {len(key)}) / secret len {len(secret)}\n")

    for label, base in ENDPOINTS:
        rest = RestClient(base_url=base, api_key=key, api_secret=secret)
        try:
            rest.get_option_positions("BTC")
            print(f"  {label:6} -> SUCCESS: key is VALID here (auth works)")
        except DeltaRestError as exc:
            code = str(exc)
            if "invalid_api_key" in code:
                verdict = "key NOT found on this platform"
            elif "ip_not_whitelisted" in code:
                verdict = "key FOUND here — only the IP is blocked (this is YOUR platform)"
            elif "Signature" in code:
                verdict = "key FOUND here — but the SECRET is wrong (this is YOUR platform)"
            else:
                verdict = f"other: {code}"
            print(f"  {label:6} -> {verdict}")
        finally:
            rest.close()


if __name__ == "__main__":
    main()
