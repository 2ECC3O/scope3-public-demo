import { useEffect, useState } from 'react';

const API = '/api/scope3';

/* One build serves three tiers. Which one the visitor gets is decided by
   what their machine can actually do, not by which zip they downloaded.

     browse  no Python backend answering. Read the page, read the numbers.
     check   backend up, no local model. Corrupt shipped data and check it.
     full    backend up. Deterministic generation and checking are available.

   A connection refused or a timeout both mean browse: that is the normal
   state for someone who opened the page without starting anything. A
   backend that answers with an HTTP error is different, and surfaces as
   `error` rather than degrading quietly, because a silent degrade looks
   exactly like a working browse tier and would hide a broken install. */
export function useBackend() {
  const [tier, setTier] = useState('loading');
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      let profile;
      try {
        // ponytail: AbortSignal.timeout is native, so no timeout wrapper of
        // our own. Cold hardware profiling can take several seconds; a
        // missing local server still fails immediately with connection refused.
        const res = await fetch(`${API}/engine-profile`, {
          signal: AbortSignal.timeout(8000),
        });
        if (!res.ok) {
          // Three ways to mean "there is no API here", none of them a fault:
          //   404       a plain static server is hosting the page (browse tier)
          //   502/504   a dev proxy with nothing behind it
          //   503       a server that is up but not serving yet
          // Anything else non-OK is a backend that answered and is unhappy,
          // which earns the error banner rather than a silent downgrade.
          const unreachable = [404, 502, 503, 504].includes(res.status);
          if (!cancelled) {
            setTier(unreachable ? 'browse' : 'error');
            if (!unreachable) {
              setDetail(`The server answered /engine-profile with HTTP ${res.status}.`);
            }
          }
          return;
        }
        profile = await res.json();
      } catch {
        if (!cancelled) setTier('browse');
        return;
      }

      // Ollama is optional; discover it only to describe the local capability.
      let models = [];
      try {
        const res = await fetch(`${API}/ollama-models`, {
          signal: AbortSignal.timeout(2500),
        });
        if (res.ok) models = (await res.json()).models || [];
      } catch {
        // Ollama being unreachable is not an error, it is the `check` tier.
      }

      if (cancelled) return;
      setTier('full');
      setDetail(models.length ? profile?.reason || null : 'No Ollama model detected; deterministic operation remains available.');
    })();

    return () => { cancelled = true; };
  }, []);

  return { tier, detail };
}
