#!/usr/bin/env python3
"""turnstile.py — click a Cloudflare Turnstile checkbox via a trusted camofox click (no solver service).

Cloudflare Turnstile is a reCAPTCHA-v2-class checkbox challenge. Like recaptcha.py for Google, this is
a sanctioned in-browser auto-solve (references/captcha-policy.md, "Sanctioned exception 3"): click the
checkbox, wait for Cloudflare to issue a token, report PASS/FAIL. On FAIL the caller falls through to
the halt-and-message procedure in that policy; this script never retries in a loop.

MECHANISM (verified live 2026-11-05 on peet.ws/turnstile-test/managed.html: token EMPTY -> issued in <3s):
The challenge iframe sits inside a CLOSED shadow root, so neither `document.querySelectorAll('iframe')`
nor cfx `frameSelector` (eval-frame / click-frame) can see it. The widget's own host box
(`.cf-turnstile` / `[data-sitekey]`) IS visible, though, and the checkbox is drawn at a fixed offset
inside it. So: scroll the host into view, read its rect, and send a TRUSTED `click-xy` at the checkbox
position. Success = the hidden `cf-turnstile-response` field gets a NEW token (never page text).

USAGE (from shell):
  CFX_TAB=... uv run python utils/turnstile.py
      Prints: PASS | PASS:already-solved | FAIL:<reason>
  Exit: 0 PASS · 1 FAIL · 2 not on a Turnstile page

LIBRARY USE (import into a board driver):
  import turnstile
  if turnstile.present():
      print("Turnstile ->", turnstile.click())   # never raises
      # if FAIL, record cooldown and halt per policy

Known limits: needs a visible `.cf-turnstile` / `[data-sitekey]` host box. Invisible Turnstile (no box)
and pages where the host box is absent return FAIL. A distrusted fingerprint can make Cloudflare issue
no token even after a real click -> FAIL:no token within Ns.
"""

import os
import subprocess
import sys
import time

# cfx.py / cfx.sh (camofox helpers) live in the job-application repo and are not vendored here.
HERE = os.environ.get("CFX_SCRIPTS", "/volume1/py/job-application/sites/_common/scripts")
sys.path.insert(0, HERE)
import cfx  # noqa: E402

# Selector for the widget's host element (the visible box that wraps the closed-shadow iframe).
_HOST = ".cf-turnstile, [data-sitekey], [data-turnstile-sitekey]"
# The checkbox is drawn this far in from the widget's left edge (Cloudflare's standard widget lays it
# out ~30px in, vertically centred). Capped for narrow widgets.
_CHECKBOX_DX = 30
# Height of the area this engine actually paints and delivers mouse events to (screenshots are
# 1280x720 even though window.innerHeight reports 1021). A click below this lands in dead space.
_PAINTED_H = 720
_TOKEN_JS = "(function(){var e=document.querySelector('[name=cf-turnstile-response],[name*=turnstile-response],[name*=cf-turnstile]');return e?(e.value||''):null;})()"


def _ev(expr, tries=3):
    """cfx.evaluate with retries; returns None on persistent failure (never raises)."""
    for _ in range(tries):
        try:
            r = cfx.evaluate(expr)
            if r is not None:
                return r
        except Exception:  # noqa: BLE001 - cfx.evaluate is flaky; callers treat None as 'unknown'
            pass
        time.sleep(1)
    return None


def _token():
    """-> '' (field present, unsolved), '<token>' (solved), or None (no field / unreadable)."""
    return _ev(_TOKEN_JS)


def _rect_raw():
    """Widget host rect in viewport px as (x, y, w, h, viewport_h), or None."""
    raw = _ev(
        "(function(){var d=document.querySelector(%r);if(!d)return '';"
        "var r=d.getBoundingClientRect();if(r.width<20||r.height<20)return '';"
        "return Math.round(r.x)+' '+Math.round(r.y)+' '+Math.round(r.width)+' '+Math.round(r.height)"
        "+' '+window.innerHeight;})()" % _HOST
    )
    if not raw or not isinstance(raw, str):
        return None
    try:
        x, y, w, h, vh = (int(v) for v in raw.split())
        return x, y, w, h, vh
    except ValueError:
        return None


def _wheel(direction, amount):
    """Real wheel-event scroll via cfx.sh (JS scrollIntoView does NOT move this engine's viewport)."""
    sh = os.path.join(HERE, "cfx.sh")
    try:
        subprocess.run(["bash", sh, "scroll", direction, str(amount)], capture_output=True, text=True, timeout=40, env=dict(os.environ))
    except Exception:  # noqa: BLE001
        pass


def _host_rect():
    """Return the widget host rect (x, y, w, h) ONLY if it is inside the painted viewport, else None.

    This engine reports innerHeight=1021 but paints/receives mouse events in only ~720px, and neither
    scrollIntoView, window.scrollTo nor wheel/End scrolling moves a page whose layout "fits" that
    taller viewport (verified: scrollY stays 0). So a widget below _PAINTED_H cannot be clicked from
    here. Try real wheel scrolling once (works on pages that genuinely overflow); if it still isn't in
    the painted area, return None so click() reports it instead of clicking dead space.
    """
    for attempt in range(3):
        r = _rect_raw()
        if not r:
            return None
        x, y, w, h, vh = r
        painted = vh if vh <= _PAINTED_H else _PAINTED_H
        if y >= 0 and y + h <= painted:
            return x, y, w, h
        if attempt == 2:
            break
        delta = y - (painted - h) // 2
        _wheel("down" if delta > 0 else "up", max(60, min(abs(delta), 600)))
        time.sleep(0.8)
    return None


def present():
    """True if a Turnstile widget appears to be on the page."""
    return bool(_ev("(function(){return !!(document.querySelector(%r)||document.querySelector('iframe[src*=\"challenges.cloudflare.com\"]')||document.querySelector('[name*=cf-turnstile]'));})()" % _HOST))


def _trusted_click(x, y):
    """Trusted mouse click at viewport coords via cfx.sh click-xy. -> True if the engine accepted it."""
    sh = os.path.join(HERE, "cfx.sh")
    try:
        out = subprocess.run(["bash", sh, "click-xy", str(x), str(y)], capture_output=True, text=True, timeout=40, env=dict(os.environ))
    except Exception:  # noqa: BLE001
        return False
    return '"ok":true' in (out.stdout or "")


def click(timeout=30):
    """Click the Turnstile checkbox once and wait for a NEW token. Never raises.

    -> 'PASS' | 'PASS:already-solved' | 'FAIL:<reason>'
    """
    try:
        if not present():
            return "FAIL:not on a turnstile page"
        if _token():  # a token already exists: solved without us -> do not click
            return "PASS:already-solved"
        rect = _host_rect()
        if not rect:
            return "FAIL:turnstile host box missing or below the %dpx painted area (page cannot be scrolled in this engine)" % _PAINTED_H
        x, y, w, h = rect
        cx = x + min(_CHECKBOX_DX, max(12, w // 6))
        cy = y + h // 2
        if not _trusted_click(cx, cy):
            return "FAIL:click-xy rejected by engine"
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            if _token():
                return "PASS"
        return "FAIL:clicked but no token within %ds (fingerprint distrusted or checkbox missed)" % timeout
    except Exception as e:  # noqa: BLE001 - contract: never raise into a driver
        return "FAIL:%s" % type(e).__name__


def main():
    if not present():
        print("FAIL:not on a turnstile page")
        return 2
    result = click()
    print(result)
    return 0 if result.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
