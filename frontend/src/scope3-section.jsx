// ponytail: functional port of the Streamlit "Scope 3 AI Auditor" app, not a
// pixel-match of its liquid-glass theme. Uses the site's existing CSS vars
// (--text, --surface-rgb, etc. from src/index.css) + Tailwind utilities only.
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Play, Loader2, CheckCircle2, XCircle, Circle, Ban,
  Factory, ShieldCheck, BarChart3, Leaf, Bot, Zap,
} from 'lucide-react';

const API_BASE = '/api/scope3';
const PUBLIC_DEMO = import.meta.env.VITE_PUBLIC_DEMO === '1';

/* ── Job polling: POST to start a pipeline step, then poll /status ──── */
function useJobPolling() {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // {state, percent, current_stage, log_tail}
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async id => {
    const res = await fetch(`${API_BASE}/status/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Could not read job (HTTP ${res.status})`);
    setStatus(data);
    if (['done', 'error', 'cancelled', 'timed_out'].includes(data.state)) stopPolling();
  }, [stopPolling]);

  const follow = useCallback(id => {
    setJobId(id);
    poll(id).catch(e => setError(String(e.message || e)));
    timerRef.current = setInterval(() => poll(id).catch(e => {
      setError(String(e.message || e)); stopPolling();
    }), 1200);
  }, [poll, stopPolling]);

  useEffect(() => {
    fetch(`${API_BASE}/active`).then(async r => {
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `Could not restore active job (HTTP ${r.status})`);
      return data;
    }).then(data => {
      if (!data.active_job_id) return;
      follow(data.active_job_id);
    }).catch(e => setError(String(e.message || e)));
  }, [follow]);

  const start = useCallback(async (endpoint, body) => {
    setError(null);
    setStarting(true);
    setStatus(null);
    stopPolling();
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        if (res.status === 409 && data.active_job_id) {
          follow(data.active_job_id);
          return;
        }
        throw new Error(data.error || `Failed to start job (HTTP ${res.status})`);
      }
      follow(data.job_id);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setStarting(false);
    }
  }, [follow, stopPolling]);

  const cancel = useCallback(async () => {
    if (!jobId) return;
    const res = await fetch(`${API_BASE}/cancel/${jobId}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Could not cancel job (HTTP ${res.status})`);
    await poll(jobId);
  }, [jobId, poll]);

  return { jobId, status, starting, error, start, cancel };
}

/* ── Discover locally installed Ollama models (mirrors the original */
/* Streamlit sidebar's model auto-detection) ─────────────────────────── */
function useOllamaModels() {
  const [models, setModels] = useState([]);
  const [best, setBest] = useState(null);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/ollama-models`)
      .then(r => r.json())
      .then(d => {
        if (cancelled) return;
        setModels(d.models || []);
        setBest(d.best || null);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, []);
  return { models, best, loaded };
}

/* ── Fetch the hardware-based engine recommendation for Verify ───────── */
function useEngineProfile() {
  const [recommended, setRecommended] = useState(null);
  const [available, setAvailable] = useState([]);
  const [reason, setReason] = useState('');
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/engine-profile`)
      .then(r => r.json())
      .then(d => {
        if (cancelled) return;
        setRecommended(d.recommended || null);
        setAvailable(d.available || []);
        setReason(d.reason || '');
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, []);
  return { recommended, available, reason, loaded };
}

/* ── Ollama LLM vs Fast Mode toggle, shared by Generate + Verify ──────── */
function OllamaControl({ enabled, onToggle, model, onModelChange, models, best, loaded }) {
  return (
    <div className="rounded-[var(--radius)] p-4 mb-2" style={{ border: `1px solid ${border}`, background: cardBg }}>
      <label className="flex items-center gap-2 text-xs font-semibold cursor-pointer" style={{ color: 'var(--text)' }}>
        <input
          type="checkbox"
          checked={enabled}
          onChange={e => onToggle(e.target.checked)}
          className="accent-current"
        />
        Use Ollama LLM
      </label>
      <p className="text-[11px] mt-1" style={{ color: 'var(--text-faint)' }}>
        Enable optional local-model assistance while generating waste-code mappings.
      </p>

      {enabled && (
        loaded && models.length > 0 ? (
          <div className="mt-2">
            <SelectField
              label="Ollama model"
              value={model === 'placeholder' ? (best || models[0]) : model}
              onChange={onModelChange}
              options={models.map(m => ({ value: m, label: m }))}
            />
          </div>
        ) : loaded ? (
          <p className="text-[11px] mt-2" style={{ color: 'var(--warn)' }}>
            No local Ollama models detected. Make sure Ollama is installed and models are downloaded, or turn this off to use Fast Mode.
          </p>
        ) : (
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-faint)' }}>Checking for local Ollama models...</p>
        )
      )}

      <p className="text-[11px] mt-2 inline-flex items-center gap-1.5" style={{ color: 'var(--text-faint)' }}>
        {enabled && model !== 'placeholder' ? (
          <><Bot size={12} /> LLM Mode: Ollama <code>{model}</code> will map data intelligently.</>
        ) : (
          <><Zap size={12} /> Fast Mode: using keyword/context matching. No Ollama required.</>
        )}
      </p>
    </div>
  );
}

/* ── Fetch the artifact listing whenever a job finishes ──────────────── */
function useStepArtifacts(job) {
  const [artifacts, setArtifacts] = useState([]);
  useEffect(() => {
    if (job.status?.state !== 'done') return;
    let cancelled = false;
    fetch(`${API_BASE}/artifacts?job_id=${encodeURIComponent(job.jobId)}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) setArtifacts(d.artifacts || []); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [job.status?.state, job.jobId]);
  return artifacts;
}

const border = 'rgba(var(--surface-rgb),0.1)';
const cardBg = 'rgba(var(--surface-rgb),0.03)';

function ProgressBar({ percent, state }) {
  const pct = Math.max(0, Math.min(100, percent || 0));
  const color = state === 'error' ? 'var(--err)' : 'var(--accent)';
  return (
    <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'rgba(var(--surface-rgb),0.08)' }}>
      <div
        className="h-full rounded-full transition-all duration-300"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

function StatusBadge({ state }) {
  const map = {
    idle: { icon: Circle, label: 'Idle', color: 'var(--text-faint)' },
    running: { icon: Loader2, label: 'Running', color: 'var(--accent)', spin: true },
    done: { icon: CheckCircle2, label: 'Done', color: 'var(--accent)' },
    error: { icon: XCircle, label: 'Failed', color: 'var(--err)' },
    cancelled: { icon: Ban, label: 'Cancelled', color: 'var(--warn)' },
    timed_out: { icon: XCircle, label: 'Time limit reached', color: 'var(--err)' },
  };
  const s = map[state] || map.idle;
  const Icon = s.icon;
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold" style={{ color: s.color }}>
      <Icon size={13} className={s.spin ? 'animate-spin' : ''} />
      {s.label}
    </span>
  );
}

function LogTail({ lines }) {
  if (!lines || lines.length === 0) return null;
  return (
    <details className="mt-3 rounded-[var(--radius)]" style={{ border: `1px solid ${border}`, background: cardBg }}>
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
        Terminal output ({lines.length} lines)
      </summary>
      <pre
        className="px-3 pb-3 text-[11px] leading-relaxed overflow-x-auto max-h-64 overflow-y-auto"
        style={{ color: 'var(--text-faint)', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }}
      >
        {lines.join('\n')}
      </pre>
    </details>
  );
}

// ponytail: naive comma split for CSV preview (no quoted-comma handling) -
// fine for this pipeline's synthetic numeric/categorical output; swap to a
// real CSV parser if a data column ever contains embedded commas.
function CsvPreview({ path, jobId }) {
  const [rows, setRows] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/artifact/${jobId}/${path}`)
      .then(r => {
        if (!r.ok) throw new Error(`preview failed with HTTP ${r.status}`);
        return r.text();
      })
      .then(text => {
        if (cancelled) return;
        const lines = text.trim().split(/\r?\n/).slice(0, 16);
        setRows(lines.map(l => l.split(',').slice(0, 12)));
      })
      .catch(() => { if (!cancelled) setRows([['(failed to load preview)']]); });
    return () => { cancelled = true; };
  }, [path, jobId]);

  if (!rows) return <div className="text-xs" style={{ color: 'var(--text-faint)' }}>Loading preview...</div>;
  const [header, ...body] = rows;
  return (
    <div className="rounded-[var(--radius)] overflow-x-auto mb-3" style={{ border: `1px solid ${border}` }}>
      <table className="text-xs w-full border-collapse">
        <thead>
          <tr style={{ background: cardBg }}>
            {header.map((h, i) => (
              <th key={i} className="px-3 py-2 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((r, ri) => (
            <tr key={ri} style={{ borderTop: `1px solid ${border}` }}>
              {r.map((c, ci) => (
                <td key={ci} className="px-3 py-1.5 whitespace-nowrap" style={{ color: 'var(--text)' }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-3 py-1.5 text-[10px]" style={{ color: 'var(--text-faintest)', background: cardBg }}>{path} (first 15 rows and 12 columns)</div>
    </div>
  );
}

function ImageGallery({ paths, jobId }) {
  if (paths.length === 0) return null;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {paths.map(p => (
        <div key={p} className="rounded-[var(--radius)] overflow-hidden" style={{ border: `1px solid ${border}` }}>
          <img src={`${API_BASE}/artifact/${jobId}/${p}`} alt={p} className="w-full h-auto block" loading="lazy" />
          <div className="px-3 py-2 text-[11px] truncate" style={{ color: 'var(--text-faint)' }}>{p}</div>
        </div>
      ))}
    </div>
  );
}

function NumberField({ label, value, onChange, min, max }) {
  return (
    <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={e => onChange(Number(e.target.value))}
        className="px-2.5 py-1.5 rounded-[var(--radius-sm)] text-sm outline-none"
        style={{ border: `1px solid ${border}`, background: 'var(--bg)', color: 'var(--text)', width: '110px' }}
      />
    </label>
  );
}

function CheckboxField({ label, hint, checked, onChange }) {
  return (
    <label className="flex flex-col gap-1 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
      <span className="inline-flex items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          className="accent-current"
        />
        {label}
      </span>
      {hint ? <span className="text-[11px]" style={{ color: 'var(--text-faintest)' }}>{hint}</span> : null}
    </label>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
      {label}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="px-2.5 py-1.5 rounded-[var(--radius-sm)] text-sm outline-none"
        style={{ border: `1px solid ${border}`, background: 'var(--bg)', color: 'var(--text)' }}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

function ErrorTypesSelect({ value, onChange }) {
  const options = [
    { value: 'all', label: 'All Errors' },
    { value: 'random', label: 'Random Mix' },
    { value: 'none', label: 'None (Pristine)' },
    { value: 'true_random_error', label: 'Enigma Scramble' },
    { value: 'fat_finger', label: 'Fat Finger' },
    { value: 'dropped_zero', label: 'Dropped Zero' },
    { value: 'unit_conversion', label: 'Unit Conversion (/1000)' },
    { value: 'scale_up_1000', label: 'Scale Up (*1000)' },
    { value: 'keyboard_mistype', label: 'Keyboard Mistype' },
    { value: 'lbs_to_kg_confusion', label: 'Lbs to Kg Confusion' },
    { value: 'currency_confusion', label: 'Currency Confusion' },
    { value: 'repeated_digits', label: 'Repeated Digits' },
    { value: 'off_by_one_digit', label: 'Off by One Digit' },
    { value: 'random_noise_high', label: 'Random Noise (High)' },
    { value: 'random_noise_low', label: 'Random Noise (Low)' },
    { value: 'accidental_zero', label: 'Accidental Zero' },
    { value: 'negative_value', label: 'Negative Value' },
  ];
  
  const selected = value.split(',').map(s => s.trim()).filter(Boolean);

  return (
    <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
      Error Types (Ctrl+Click to pick)
      <select
        multiple
        value={selected}
        onChange={e => {
          const vals = Array.from(e.target.selectedOptions, option => option.value);
          if (vals.includes('all')) onChange('all');
          else if (vals.includes('random')) onChange('random');
          else if (vals.includes('none')) onChange('none');
          else onChange(vals.join(', '));
        }}
        className="px-2.5 py-1.5 rounded-[var(--radius-sm)] text-sm outline-none"
        style={{ border: `1px solid rgba(var(--surface-rgb),0.1)`, background: 'var(--bg)', color: 'var(--text)', height: '110px', width: '220px' }}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

/* ── Extract the illustrative Step 4 metrics from stdout ─── */
function parseMaterialityMetrics(logLines) {
  const text = (logLines || []).join('\n');
  const grab = (pattern, cast = Number) => {
    const m = text.match(pattern);
    return m ? cast(m[1].replace(/,/g, '')) : null;
  };
  return {
    total: grab(/Total Companies Evaluated:\s+([\d,]+)/),
    withErrors: grab(/Companies with >1% aggregate emissions discrepancy:\s+([\d,]+)/),
    trueAvg: grab(/Average Pristine-Based Illustrative Score:\s+([\d.]+)/, parseFloat),
    reportedAvg: grab(/Average Reported Illustrative Score:\s+([\d.]+)/, parseFloat),
    correctedAvg: grab(/Average Model-Output Illustrative Score:\s+([\d.]+)/, parseFloat),
    higher: grab(/scored HIGHER after correction:\s+([\d,]+)/),
    lower: grab(/scored LOWER\s+after correction:\s+([\d,]+)/),
  };
}

function StatTile({ label, value }) {
  return (
    <div className="rounded-[var(--radius-sm)] px-3 py-2.5 text-center" style={{ border: `1px solid ${border}`, background: cardBg }}>
      <div className="text-lg font-bold" style={{ color: 'var(--text)' }}>{value ?? '-'}</div>
      <div className="text-[10px] uppercase tracking-wide mt-0.5" style={{ color: 'var(--text-faint)' }}>{label}</div>
    </div>
  );
}

/* ── One pipeline-step tab: run button, progress, log, artifacts ─────── */
function StepPanel({ meta, job, onRun, renderResults, busy, unavailable }) {
  const Icon = meta.icon;
  const state = job.status?.state || 'idle';
  const running = state === 'running' || job.starting;

  return (
    <div className="rounded-[var(--radius)] p-6" style={{ border: `1px solid ${border}`, background: cardBg }}>
      <div className="flex items-start justify-between gap-4 mb-4 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="rounded-[var(--radius-sm)] p-2.5" style={{ background: 'rgba(var(--accent-rgb),0.12)' }}>
            <Icon size={20} style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h3 className="font-display text-lg font-bold" style={{ color: 'var(--text)' }}>{meta.title}</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>{meta.description}</p>
          </div>
        </div>
        <StatusBadge state={state} />
      </div>

      <div className="flex flex-wrap items-end gap-4 mb-4">
        {meta.fields}
        <button
          onClick={onRun}
          disabled={busy || unavailable}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-[var(--radius-sm)] text-sm font-semibold transition-opacity"
          style={{
            background: 'var(--accent)',
            color: 'var(--on-accent)',
            opacity: busy || unavailable ? 0.6 : 1,
            cursor: busy || unavailable ? 'not-allowed' : 'pointer',
          }}
        >
          {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {running ? 'Running...' : busy ? 'Another run is active' : unavailable || `Run ${meta.title}`}
        </button>
        {running ? (
          <button type="button" onClick={job.cancel} className="px-4 py-2 rounded-lg text-sm font-semibold" style={{ border: `1px solid ${border}` }}>
            Cancel
          </button>
        ) : null}
      </div>

      {job.error && (
        <p className="text-xs mb-3" style={{ color: 'var(--err)' }}>{job.error}</p>
      )}

      {(job.status || job.starting) && (
        <div className="mb-1">
          <div className="flex justify-between text-[11px] mb-1" style={{ color: 'var(--text-faint)' }}>
            <span>{job.status?.current_stage || 'Starting...'}</span>
            <span>{job.status?.percent ?? 0}%</span>
          </div>
          <ProgressBar percent={job.status?.percent} state={state} />
        </div>
      )}

      <LogTail lines={job.status?.log_tail} />

      {state === 'done' && (
        <div className="mt-4">{renderResults()}</div>
      )}
    </div>
  );
}

export default function Scope3Section({ enabled = true }) {
  const [companies, setCompanies] = useState(3);
  const [products, setProducts] = useState(20);
  const [errorRate, setErrorRate] = useState(15);
  const [errorTypes, setErrorTypes] = useState('random');
  const [useOllama, setUseOllama] = useState(false);
  const [ollamaModel, setOllamaModel] = useState('placeholder');
  const [mode, setMode] = useState('max_accuracy');
  const [scope, setScope] = useState('total');
  const [shapMode, setShapMode] = useState('production');
  const [hardware] = useState('1');
  const [engine, setEngine] = useState('auto');

  const ollama = useOllamaModels();
  const engineProfile = useEngineProfile();

  // Once models are known and the user enables Ollama, default the selection
  // to the best-detected model (mirrors the original sidebar's selectbox
  // defaulting to best_detected_model) so the UI never shows a stale
  // "placeholder" state while functionally already using a real model.
  useEffect(() => {
    if (useOllama && ollamaModel === 'placeholder' && ollama.models.length > 0) {
      setOllamaModel(ollama.best || ollama.models[0]);
    }
  }, [useOllama, ollama.models, ollama.best, ollamaModel]);

  // Effective model sent to the backend: the real model name only when the
  // toggle is on AND a model is actually selected, otherwise "placeholder"
  // (= Fast Mode / bypass LLM, matching the pipeline scripts' own default).
  const effectiveOllamaModel = useOllama && ollamaModel !== 'placeholder' ? ollamaModel : 'placeholder';

  const job = useJobPolling();
  const busy = job.starting || job.status?.state === 'running';
  const forStep = step => job.status?.step === step ? job : { ...job, status: null, error: null };
  const generateJob = forStep('generate');
  const verifyJob = forStep('verify');
  const accuracyJob = forStep('accuracy');
  const materialityJob = forStep('materiality');

  const [completed, setCompleted] = useState(() => {
    try { return JSON.parse(localStorage.getItem('scope3-completed-jobs') || '{}'); }
    catch { return {}; }
  });
  useEffect(() => {
    const step = job.status?.step;
    if (job.status?.state !== 'done' || !step || !job.jobId) return;
    setCompleted(previous => {
      const next = { ...previous, [step]: job.jobId };
      const downstream = {
        generate: ['verify', 'accuracy', 'materiality'],
        verify: ['accuracy', 'materiality'],
        accuracy: ['materiality'],
      };
      for (const stale of downstream[step] || []) delete next[stale];
      localStorage.setItem('scope3-completed-jobs', JSON.stringify(next));
      return next;
    });
  }, [job.status?.state, job.status?.step, job.jobId]);

  const generateArtifacts = useStepArtifacts(generateJob);
  const verifyArtifacts = useStepArtifacts(verifyJob);
  const accuracyArtifacts = useStepArtifacts(accuracyJob);
  const demoJob = forStep('demo');
  const demoArtifacts = useStepArtifacts(demoJob);

  return (
    <div className="max-w-4xl mx-auto">
      <OllamaControl
        enabled={useOllama}
        onToggle={setUseOllama}
        model={ollamaModel}
        onModelChange={setOllamaModel}
        models={ollama.models}
        best={ollama.best}
        loaded={ollama.loaded}
      />
      <p className="text-[11px] mb-6" style={{ color: 'var(--text-faintest)' }}>
        Applies only to optional waste-code assistance during Generate. Verification does not call a language model.
      </p>

      <div className="flex flex-col gap-6">
        <StepPanel
          meta={{ title: 'Bundled check', icon: ShieldCheck, description: PUBLIC_DEMO ? 'Generate 36 rows from invented teaching inputs, then check and score them without Ollama.' : 'Check and score the fixed 36-row synthetic example without Ollama.', fields: null }}
          job={forStep('demo')}
          busy={busy}
          unavailable={!enabled && 'Start local server'}
          onRun={() => job.start(`${API_BASE}/demo`, {})}
          renderResults={() => <div className="text-xs">The real verifier and scorer completed. {demoArtifacts.map(a => <a key={a.path} className="block underline mt-1" href={`${API_BASE}/artifact/${demoJob.jobId}/${a.path}`}>{a.path}</a>)}</div>}
        />
        <StepPanel
          meta={{
            title: 'Generate',
            icon: Factory,
            description: 'Synthesize multi-product Scope 3 supply-chain data with injected reporting anomalies.',
            fields: (
              <>
                <NumberField label="Companies" value={companies} min={1} max={3} onChange={setCompanies} />
                <NumberField label="Products / company" value={products} min={1} max={20} onChange={setProducts} />
                <NumberField label="Error Rate (%)" value={errorRate} min={0} max={100} onChange={setErrorRate} />
                <ErrorTypesSelect value={errorTypes} onChange={setErrorTypes} />
              </>
            ),
          }}
          job={generateJob}
          busy={busy}
          unavailable={!enabled && 'Start local server'}
          onRun={() => generateJob.start(`${API_BASE}/generate`, { companies, products, ollama_model: effectiveOllamaModel, error_rate: errorRate / 100.0, error_types: errorTypes })}
          renderResults={() => {
            const csvs = generateArtifacts.filter(a => a.path.includes('/generated company/') && a.path.endsWith('.csv') && !a.path.includes('_AI_corrected'));
            return csvs.length
              ? <div>{csvs.slice(0, 2).map(a => <CsvPreview key={a.path} path={a.path} jobId={generateJob.jobId} />)}</div>
              : <p className="text-xs" style={{ color: 'var(--text-faint)' }}>No generated files found yet.</p>;
          }}
        />

        <StepPanel
          meta={{
            title: 'Verify',
            icon: ShieldCheck,
            description: 'Run statistical anomaly models with deterministic mass-balance, ratio, range, and identity checks.',
            fields: (
              <>
                <NumberField label="Companies" value={companies} min={1} max={3} onChange={setCompanies} />
                <SelectField label="Mode" value={mode} onChange={setMode} options={[
                  { value: 'speed', label: 'Speed' },
                  { value: 'accuracy', label: 'Accuracy' },
                  { value: 'max_accuracy', label: 'Max Accuracy' },
                ]} />
                <SelectField label="Scope" value={scope} onChange={setScope} options={[
                  { value: 'fast', label: 'Fast (4 summaries)' },
                  { value: 'total', label: 'Total (all columns)' },
                ]} />
                <SelectField label="SHAP" value={shapMode} onChange={setShapMode} options={[
                  { value: 'researcher', label: 'Researcher (with SHAP)' },
                  { value: 'production', label: 'Production (no SHAP)' },
                ]} />
                <SelectField label="Engine" value={engine} onChange={setEngine} options={[
                  { value: 'auto', label: engineProfile.recommended ? `Auto (Recommended: ${engineProfile.recommended})` : 'Auto' },
                  { value: 'heuristic', label: 'Heuristic' },
                  { value: 'conditional', label: `Conditional${engineProfile.available.includes('conditional') ? '' : ' (unavailable)'}` },
                  { value: 'full', label: `Full (PySR)${engineProfile.available.includes('full') ? '' : ' (unavailable)'}` },
                ]} />
                {engineProfile.reason ? (
                  <p className="text-[11px] w-full" style={{ color: 'var(--text-faint)' }}>{engineProfile.reason}</p>
                ) : null}
              </>
            ),
          }}
          job={verifyJob}
          busy={busy}
          unavailable={!enabled ? 'Start local server' : !completed.generate && 'Generate first'}
          onRun={() => verifyJob.start(`${API_BASE}/verify`, {
            companies, mode, scope, shap_mode: shapMode, hardware, ollama_model: effectiveOllamaModel, engine, source_job_id: completed.generate,
          })}
          renderResults={() => {
            const shapImages = verifyArtifacts.filter(a => a.type === 'image' && a.path.includes('/ai corrected/dashboards/'));
            return shapImages.length
              ? <ImageGallery paths={shapImages.map(a => a.path)} jobId={verifyJob.jobId} />
              : <p className="text-xs" style={{ color: 'var(--text-faint)' }}>No SHAP plots produced (production mode or no anomalies detected).</p>;
          }}
        />

        <StepPanel
          meta={{
            title: 'Accuracy',
            icon: BarChart3,
            description: 'Generate 6-panel accuracy diagnostics comparing pristine, messy, and AI-corrected data.',
            fields: <NumberField label="Companies" value={companies} min={1} max={3} onChange={setCompanies} />,
          }}
          job={accuracyJob}
          busy={busy}
          unavailable={!enabled ? 'Start local server' : !completed.verify && 'Verify first'}
          onRun={() => accuracyJob.start(`${API_BASE}/accuracy`, { companies, source_job_id: completed.verify })}
          renderResults={() => {
            const plots = accuracyArtifacts.filter(a => a.type === 'image' && a.path.includes('/accuracy test/'));
            return plots.length
              ? <ImageGallery paths={plots.map(a => a.path)} jobId={accuracyJob.jobId} />
              : <p className="text-xs" style={{ color: 'var(--text-faint)' }}>No diagnostic plots found yet.</p>;
          }}
        />

        <StepPanel
          meta={{
            title: 'Materiality',
            icon: Leaf,
            description: 'Calculate an illustrative materiality score for the synthetic inventory.',
            fields: <NumberField label="Companies" value={companies} min={1} max={3} onChange={setCompanies} />,
          }}
          job={materialityJob}
          busy={busy}
          unavailable={!enabled ? 'Start local server' : !completed.accuracy && 'Score accuracy first'}
          onRun={() => materialityJob.start(`${API_BASE}/materiality`, { companies, source_job_id: completed.accuracy })}
          renderResults={() => {
            const m = parseMaterialityMetrics(materialityJob.status?.log_tail);
            return (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <StatTile label="Companies Evaluated" value={m.total} />
                <StatTile label="Aggregate difference >1%" value={m.withErrors} />
                <StatTile label="Pristine-based score" value={m.trueAvg} />
                <StatTile label="Illustrative reported score" value={m.reportedAvg} />
                <StatTile label="Model-output score" value={m.correctedAvg} />
                <StatTile label="Scored Higher / Lower" value={m.higher != null ? `${m.higher} / ${m.lower}` : null} />
              </div>
            );
          }}
        />
      </div>
    </div>
  );
}
