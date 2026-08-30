# -*- coding: utf-8 -*-
"""OpenAlex request helper.

Reads two optional environment variables:
  OPENALEX_MAILTO   e-mail address for the OpenAlex polite pool (recommended)
  OPENALEX_API_KEY  premium API key, if you have one (not required)

With neither set, requests still work against the public anonymous pool.
"""
import os

MAILTO = os.environ.get("OPENALEX_MAILTO", "")


def get_api_key():
    key = os.environ.get("OPENALEX_API_KEY")
    return key or None


def params(extra=None):
    p = {}
    if MAILTO:
        p["mailto"] = MAILTO
    key = get_api_key()
    if key:
        p["api_key"] = key
    if extra:
        p.update(extra)
    return p
