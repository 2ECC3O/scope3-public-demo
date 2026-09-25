import { useState } from 'react';
import {
  Factory, ShieldCheck, BarChart3, Leaf,
  ArrowRight, Github, TriangleAlert, Copy, Check, Image as ImageIcon,
} from 'lucide-react';
import Scope3Section from './scope3-section.jsx';
import { useBackend } from './useBackend.js';
import demo from './demo-results.json';
const PUBLIC_DEMO = import.meta.env.VITE_PUBLIC_DEMO === '1';

const REPO = '';

/* Generated from the bundled run by tools/build_demo_results.py. */
const METRICS = [
  { value: demo.rates_percent.tp_flag_rate, label: 'Eligible remaining errors flagged', denominator: `${demo.counts.eligible_flagged_tp} of ${demo.counts.should_flag}` },
  { value: demo.rates_percent.detection_accuracy, label: 'Numeric cells classified correctly', denominator: `${demo.counts.tp + demo.counts.tn} of ${demo.counts.n_cells}` },
  { value: demo.rates_percent.degraded_rate, label: 'Corrections made worse', denominator: `${demo.counts.degraded} of ${demo.counts.true_errors}` },
];

const STEPS = [
  {
    icon: Factory,
    title: 'Generate',
    body: 'Synthesises multi-product supply-chain data for three companies, then injects known reporting errors at a rate you choose.',
    span: 'md:col-span-2',
    image: true,
  },
  {
    icon: ShieldCheck,
    title: 'Verify',
    body: 'Uses statistical anomaly models alongside deterministic mass-balance, ratio, range, and identity checks.',
    span: 'md:col-span-1',
    pattern: true,
  },
  {
    icon: BarChart3,
    title: 'Accuracy',
    body: 'A separate scoring step compares reported and corrected values with the pristine synthetic reference.',
    span: 'md:col-span-1',
  },
  {
    icon: Leaf,
    title: 'Materiality',
    body: 'Calculates an illustrative materiality score so reported and corrected synthetic inventories can be compared.',
    span: 'md:col-span-2',
    wash: true,
  },
];

/* A locked tier has to say what unlocks it. Showing three greyed boxes and
   leaving the visitor to guess is the failure mode this strip exists to
   avoid, so every tier above the current one carries its own instruction. */
const TIERS = [
  {
    key: 'browse',
    title: 'Browse',
    body: PUBLIC_DEMO ? 'Read the demonstration limits. Nothing installed.' : 'Read the page and the published results. Nothing installed.',
  },
  {
    key: 'check',
    title: 'Check',
    body: 'Check the bundled synthetic example.',
    unlock: 'Unlocked by the install below.',
  },
  {
    key: 'full',
    title: 'Generate',
    body: 'Generate fresh deterministic synthetic data. A local model is optional.',
    unlock: 'Unlocked by the same core installation.',
  },
];

/* The install line has to work on a machine with no Python on it, so it
   cannot be a Python command. Each platform gets the shell it always has:
   sh on macOS and Linux, PowerShell on Windows. The installer asks which
   tier you want and only installs that much.

   Ollama is not automated by the installer, on purpose: a separate runtime
   plus a multi-gigabyte model pull is not a decision something you pasted
   into a terminal should make for you. It defers here instead, which is why
   the commands below have to exist. */
const INSTALL = [
  {
    unlocks: 'check',
    title: 'Install, or add the checker',
    note: <>{PUBLIC_DEMO ? 'From this extracted archive, set SCOPE3_SOURCE to this folder before running the installer. ' : 'From a checkout you have already reviewed, '}Install uv 0.12.13 from the <a className="underline" href="https://docs.astral.sh/uv/getting-started/installation/">official uv installation guide</a>, then run its included installer.</>,
    lines: [
      {
        os: 'macOS, Linux',
        cmd: PUBLIC_DEMO ? 'SCOPE3_SOURCE="$PWD" ./install.sh' : './install.sh',
      },
      {
        os: 'Windows',
        cmd: PUBLIC_DEMO ? '$env:SCOPE3_SOURCE=(Get-Location).Path; .\\install.ps1' : '.\\install.ps1',
      },
    ],
    footnote: 'After installation, start with .tools\\uv.exe run --no-project --python .venv\\Scripts\\python.exe -- python start.py on Windows. Normal startup does not install packages.',
  },
];

/* ── Primitives ───────────────────────────────────────────────────── */

/* IMAGE SLOTS. Both visuals are placeholders drawn from the page's own palette.
   They were hotlinked to picsum.photos, which meant a third-party request per
   visitor and visibly random imagery on a page that is job evidence.

   To drop in a real picture: put the file in frontend/public/ and replace the
   <ImageSlot .../> call with
     <img src="/your-file.jpg" alt="..." className="w-full h-full object-cover" />
   keeping the wrapper's aspect ratio. Nothing else needs to change. */
function ImageSlot({ className = '', label }) {
  return (
    <div
      className={`relative overflow-hidden flex items-center justify-center ${className}`}
      style={{
        background:
          'linear-gradient(135deg, var(--accent-wash) 0%, var(--card) 60%, var(--accent-wash) 100%)',
      }}
      role="presentation"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0"
        style={{
          backgroundImage:
            'radial-gradient(rgba(var(--accent-rgb),0.14) 1px, transparent 1px)',
          backgroundSize: '16px 16px',
        }}
      />
      <ImageIcon
        size={26}
        aria-hidden="true"
        style={{ color: 'var(--accent)', opacity: 0.5 }}
        className="relative"
      />
      {label ? (
        <span className="sr-only">{label}</span>
      ) : null}
    </div>
  );
}

function Button({ href, children, variant = 'primary', className = '' }) {
  const base =
    'inline-flex items-center justify-center gap-2 whitespace-nowrap px-5 py-2.5 ' +
    'text-[15px] font-semibold transition-all duration-200 ' +
    'active:translate-y-px';
  const styles =
    variant === 'primary'
      ? { background: 'var(--accent)', color: 'var(--on-accent)', borderRadius: 'var(--radius-sm)' }
      : {
          background: 'transparent',
          color: 'var(--text)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
        };
  return (
    <a href={href} className={`${base} ${className}`} style={styles}>
      {children}
    </a>
  );
}

function Section({ id, children, soft = false, className = '' }) {
  return (
    <section
      id={id}
      className={`px-6 py-20 md:py-28 ${className}`}
      style={soft ? { background: 'var(--bg-soft)' } : undefined}
    >
      <div className="max-w-6xl mx-auto">{children}</div>
    </section>
  );
}

/* ── Page ─────────────────────────────────────────────────────────── */

function Nav() {
  const link = 'text-[15px] transition-colors duration-200 hover:opacity-70';
  return (
    <header
      className="sticky top-0 z-40 h-[68px] flex items-center px-6"
      style={{
        background: 'color-mix(in srgb, var(--bg) 88%, transparent)',
        borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(8px)',
      }}
    >
      <nav className="max-w-6xl mx-auto w-full flex items-center gap-8">
        <a href="#top" className="font-display text-[17px] font-bold shrink-0">
          Scope 3 Auditor
        </a>
        <div className="hidden md:flex items-center gap-7 ml-auto" style={{ color: 'var(--text-secondary)' }}>
          <a href="#how" className={link}>How it works</a>
          <a href="#results" className={link}>Results</a>
          <a href="#auditor" className={link}>Open the auditor</a>
        </div>
        {!PUBLIC_DEMO && <a
          href={REPO}
          className="ml-auto md:ml-0 inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold
                     transition-all duration-200 active:translate-y-px"
          style={{
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text)',
          }}
        >
          <Github size={16} aria-hidden="true" />
          <span className="hidden sm:inline">View on GitHub</span>
        </a>}
      </nav>
    </header>
  );
}

function Hero() {
  return (
    <section id="top" className="px-6 pt-16 md:pt-24 pb-16 md:pb-20">
      <div className="max-w-6xl mx-auto grid gap-12 lg:gap-16 lg:grid-cols-[1.15fr_0.85fr] items-center">
        <div>
          <p
            className="rise text-[13px] font-semibold uppercase tracking-[0.14em] mb-5"
            style={{ color: 'var(--accent)', '--i': 0 }}
          >
            Scope 3 emissions data
          </p>
          <h1
            className="rise font-display text-4xl md:text-5xl lg:text-[3.25rem] font-bold leading-[1.08] mb-6"
            style={{ '--i': 1 }}
          >
            Find the errors in supplier emissions data.
          </h1>
          <p
            className="rise text-lg leading-relaxed max-w-[54ch] mb-9"
            style={{ color: 'var(--text-secondary)', '--i': 2 }}
          >
            {PUBLIC_DEMO ? 'Explore an offline research prototype with invented teaching inputs. Run the local checker and inspect its output.' : 'An open auditing pipeline that generates realistic Scope 3 datasets, injects known errors, and measures how many it catches.'}
          </p>
          <div className="rise flex flex-wrap gap-3" style={{ '--i': 3 }}>
            <Button href="#auditor">
              Open the auditor
              <ArrowRight size={17} aria-hidden="true" />
            </Button>
            <Button href="#how" variant="secondary">How it works</Button>
          </div>
        </div>

        <div
          className="rise relative overflow-hidden aspect-[4/3] lg:aspect-[5/4]"
          style={{
            borderRadius: 'var(--radius)',
            border: '1px solid var(--border)',
            background: 'var(--accent-wash)',
            '--i': 2,
          }}
        >
          <ImageSlot className="w-full h-full" label="Abstract emissions data flow" />
        </div>
      </div>
    </section>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        // Fails outside a secure context; the command stays selectable, so
        // there is nothing to recover from beyond not showing the tick.
        navigator.clipboard?.writeText(text).then(
          () => { setCopied(true); setTimeout(() => setCopied(false), 1500); },
          () => {},
        );
      }}
      aria-label={copied ? 'Copied' : `Copy: ${text}`}
      className="shrink-0 px-2 flex items-center transition-colors duration-200"
      style={{
        border: '1px solid var(--border)',
        borderRadius: '4px',
        color: copied ? 'var(--accent)' : 'var(--text-faint)',
        background: 'transparent',
      }}
    >
      {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
    </button>
  );
}

function InstallGuide({ tier }) {
  // Nothing to install once every tier is live, and nothing to claim while
  // detection is still in flight.
  const order = ['browse', 'check', 'full'];
  const have = order.indexOf(tier);
  if (have < 0) return null;
  const groups = INSTALL.filter(g => order.indexOf(g.unlocks) > have);
  if (!groups.length) return null;

  return (
    <div
      className="mt-4 p-6"
      style={{
        borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
        background: 'var(--card)',
      }}
    >
      <h2 className="text-sm font-bold mb-1">Turn on the rest</h2>
      <p className="text-[13px] mb-6" style={{ color: 'var(--text-faint)' }}>
        The page checked what this machine can do. Here is what unlocks the rest.
      </p>

      <div className="grid gap-7">
        {groups.map(g => (
          <div key={g.unlocks} className="min-w-0">
            <h3 className="text-[13px] font-bold mb-1">{g.title}</h3>
            <p className="text-[12.5px] leading-snug mb-3" style={{ color: 'var(--text-faint)' }}>
              {g.note}
            </p>
            <dl className="grid gap-4">
              {g.lines.map(l => (
                <div key={l.os + l.cmd} className="min-w-0">
                  <dt
                    className="text-[11px] font-semibold mb-1.5"
                    style={{ color: 'var(--text-faint)' }}
                  >
                    {l.os}
                  </dt>
                  <dd className="flex items-stretch gap-1.5">
                    <code
                      className="font-num text-[12px] px-2 py-1.5 flex-1 min-w-0 overflow-x-auto whitespace-pre"
                      style={{
                        background: 'var(--accent-wash)',
                        color: 'var(--text-secondary)',
                        borderRadius: '4px',
                      }}
                    >
                      {l.cmd}
                    </code>
                    <CopyButton text={l.cmd} />
                  </dd>
                </div>
              ))}
            </dl>
            {g.footnote ? (
              <p className="text-[12px] mt-3" style={{ color: 'var(--text-faint)' }}>
                {g.footnote}
              </p>
            ) : null}
          </div>
        ))}
      </div>

      <p className="text-[12px] mt-6" style={{ color: 'var(--text-faint)' }}>
        Reload this page afterwards and the tiers above will update.
      </p>
    </div>
  );
}

function TierStrip({ tier, detail }) {
  if (tier === 'error') {
    return (
      <div className="px-6 pb-4">
        <div
          className="max-w-6xl mx-auto flex items-start gap-3 p-4 text-sm"
          style={{
            borderRadius: 'var(--radius)',
            border: '1px solid var(--err)',
            color: 'var(--err)',
          }}
        >
          <TriangleAlert size={18} className="shrink-0 mt-0.5" aria-hidden="true" />
          <p>
            The Python server is running but answered with an error, so the auditor below will not
            work. {detail} Check the terminal you started it from.
          </p>
        </div>
      </div>
    );
  }

  const activeIndex = TIERS.findIndex(t => t.key === tier);
  const loading = tier === 'loading';

  return (
    <div className="px-6 pb-8">
      <div
        className="max-w-6xl mx-auto grid sm:grid-cols-3"
        style={{
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
          background: 'var(--card)',
          overflow: 'hidden',
        }}
        aria-busy={loading}
      >
        {TIERS.map((t, i) => {
          const on = !loading && i <= activeIndex;
          return (
            <div
              key={t.key}
              className="p-5 transition-colors duration-300"
              style={{
                background: on ? 'var(--accent-wash)' : 'transparent',
                borderLeft: i === 0 ? undefined : '1px solid var(--border)',
                opacity: loading ? 0.45 : 1,
              }}
            >
              <div className="flex items-center gap-2 mb-1.5">
                {/* Real state, not decoration: this is what the machine can do. */}
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ background: on ? 'var(--accent)' : 'var(--border)' }}
                  aria-hidden="true"
                />
                <h2 className="text-sm font-bold">{t.title}</h2>
                {on && i === activeIndex ? (
                  <span className="text-[11px] font-semibold" style={{ color: 'var(--accent)' }}>
                    available
                  </span>
                ) : null}
              </div>
              <p className="text-[13px] leading-snug" style={{ color: 'var(--text-faint)' }}>
                {t.body}
              </p>
              {!on && t.unlock ? (
                <p className="mt-2 text-[12px] leading-snug" style={{ color: 'var(--text-faint)' }}>
                  {t.unlock}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="max-w-6xl mx-auto"><InstallGuide tier={tier} /></div>
    </div>
  );
}

function Results() {
  if (PUBLIC_DEMO) return (
    <Section id="results" soft>
      <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">Run an invented-data demonstration</h2>
      <p className="max-w-[65ch]" style={{ color: 'var(--text-secondary)' }}>
        This public package generates a fresh, fixed-seed example when you select Bundled check below.
        Its factors and waste codes are invented teaching inputs, not environmental estimates.
        Download the resulting files to inspect the actual verifier and scorer output.
      </p>
      <a className="inline-block mt-6 underline" href="#auditor">Open the local auditor</a>
    </Section>
  );
  return (
    <Section id="results" soft>
      <div className="reveal grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-end">
        <div>
          <h2 className="font-display text-3xl md:text-4xl font-bold leading-tight mb-4">
            A real, reproducible tiny check.
          </h2>
          <p className="text-base leading-relaxed max-w-[52ch]" style={{ color: 'var(--text-secondary)' }}>
            These figures come from the bundled 36-row synthetic example, checked and scored by the
            actual pipeline without Ollama. They are demonstration results, not a full benchmark.
          </p>
        </div>

        <div>
          <p className="font-figure text-6xl md:text-7xl font-medium leading-none" style={{ color: 'var(--accent)' }}>
            {demo.rates_percent.fp_flag_rate}<span className="text-3xl md:text-4xl align-top">%</span>
          </p>
          <p className="mt-3 text-base font-semibold">Flags that turned out to be clean cells</p>
          <p className="mt-1 text-sm" style={{ color: 'var(--text-faint)' }}>
            {demo.counts.clean_flags} clean flags out of {demo.counts.flagged_total} total flags in this tiny run.
          </p>
        </div>
      </div>

      <div className="reveal grid gap-px mt-14 sm:grid-cols-3" style={{ background: 'var(--border)' }}>
        {METRICS.map(m => (
          <div key={m.label} className="p-6" style={{ background: 'var(--bg-soft)' }}>
            <p className="font-figure text-3xl font-medium leading-none">
              {m.value}<span className="text-lg align-top">%</span>
            </p>
            <p className="mt-2.5 text-sm font-semibold">{m.label}</p>
            <p
              className="mt-1 text-[13px]"
              style={{ color: 'var(--text-faint)' }}
            >
              {m.denominator}
            </p>
          </div>
        ))}
      </div>
      <div className="mt-3 p-4 rounded-lg" style={{ border: '1px solid var(--border)', background: 'var(--card)' }}>
        <p className="font-num text-xl font-semibold">{demo.row_reviews.rows} of {demo.row_reviews.total_rows} rows</p>
        <p className="mt-1 text-sm font-semibold">Rows needing human review</p>
        <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
          {demo.row_reviews.events} review events. {demo.row_reviews.explanation}
        </p>
      </div>
      <p className="mt-6 text-[12px] leading-relaxed" style={{ color: 'var(--text-faint)' }}>
        Source: {demo.provenance.source}; code/data manifest SHA-256 {demo.provenance.code_data_manifest_sha256}.
        Settings: {demo.provenance.settings.companies} company, {demo.provenance.settings.products} products,
        {demo.provenance.settings.rows} rows; {demo.provenance.settings.verify.join('/')}; no Ollama.
        Input SHA-256: {demo.provenance.source_sha256}. {demo.limitations}
      </p>
      <div className="mt-10 overflow-x-auto" style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
        <table className="w-full text-[12px] text-left">
          <thead style={{ background: 'var(--card)' }}>
            <tr>{['Company / product / month', 'Field', 'Reported', 'Suggested', 'Verdict and reason'].map(h => <th key={h} className="p-3">{h}</th>)}</tr>
          </thead>
          <tbody>
            {demo.examples.map(example => (
              <tr key={`${example.row}-${example.field}`} style={{ borderTop: '1px solid var(--border)' }}>
                <td className="p-3 whitespace-nowrap">{example.company}<br />{example.product}<br />{example.month}</td>
                <td className="p-3 font-num">{example.field}</td>
                <td className="p-3 font-num">{example.reported_value}</td>
                <td className="p-3 font-num">{example.suggested_correction ?? 'Review only'}</td>
                <td className="p-3"><strong>{example.verdict}</strong><br />{example.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function HowItWorks() {
  return (
    <Section id="how">
      <h2 className="reveal font-display text-3xl md:text-4xl font-bold leading-tight mb-3 max-w-[20ch]">
        Four stages, run in order.
      </h2>
      <p className="reveal text-base leading-relaxed max-w-[58ch] mb-12" style={{ color: 'var(--text-secondary)' }}>
        Each stage writes files the next one reads, so you can stop after any of them and inspect
        what it produced.
      </p>

      <div className="grid gap-4 md:grid-cols-3">
        {STEPS.map(s => (
          <article
            key={s.title}
            className={`reveal overflow-hidden flex flex-col ${s.span}`}
            style={{
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border)',
              background: s.wash ? 'var(--accent-wash)' : 'var(--card)',
              backgroundImage: s.pattern
                ? 'radial-gradient(rgba(var(--accent-rgb),0.16) 1px, transparent 1px)'
                : undefined,
              backgroundSize: s.pattern ? '14px 14px' : undefined,
            }}
          >
            {s.image ? (
              <ImageSlot className="w-full h-44" label={`${s.title} data-flow illustration`} />
            ) : null}
            <div className="p-6">
              <s.icon size={20} style={{ color: 'var(--accent)' }} aria-hidden="true" />
              <h3 className="mt-3 text-lg font-bold">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {s.body}
              </p>
            </div>
          </article>
        ))}
      </div>
    </Section>
  );
}

function Coverage() {
  return (
    <Section soft>
      <div className="reveal grid gap-12 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
        <div>
          <h2 className="font-display text-3xl md:text-4xl font-bold leading-tight mb-5 max-w-[22ch]">
            {PUBLIC_DEMO ? 'All 45 material factors are invented teaching inputs.' : '16 of the 45 materials have a real emission factor.'}
          </h2>
          <p className="text-base leading-relaxed max-w-[54ch] mb-4" style={{ color: 'var(--text-secondary)' }}>
            {PUBLIC_DEMO ? 'One factor is explicit in the invented database; the other 44 come from a deterministic hash. The run log identifies them.' : 'The other 29 fall back to a hash of the material name. That is fabricated data, and the pipeline prints which ones at every startup rather than hiding it behind an average.'}
          </p>
          <p className="text-base leading-relaxed max-w-[54ch]" style={{ color: 'var(--text-secondary)' }}>
            Anything you run here is synthetic. The checker is bound to this project's column names,
            so it will not read your own spreadsheet.
          </p>
        </div>

        <div>
          <div className="grid grid-cols-9 gap-1.5 max-w-[320px]" role="img"
               aria-label={PUBLIC_DEMO ? 'One invented database factor and 44 invented fallback factors.' : '16 of 45 materials have a published emission factor; the remaining 29 are fabricated.'}>
            {Array.from({ length: 45 }, (_, i) => (
              <span
                key={i}
                className="aspect-square"
                style={{
                  borderRadius: '3px',
                  background: i < (PUBLIC_DEMO ? 1 : 16) ? 'var(--accent)' : 'transparent',
                  border: i < (PUBLIC_DEMO ? 1 : 16) ? undefined : '1px solid var(--border)',
                }}
              />
            ))}
          </div>
          <p className="mt-4 text-[13px]" style={{ color: 'var(--text-faint)' }}>
            {PUBLIC_DEMO ? 'The filled square is an explicit invented factor. Outlines are invented hash values.' : 'Filled squares are published Thai factors. Outlines are fabricated.'}
          </p>
        </div>
      </div>
    </Section>
  );
}

function Auditor({ tier }) {
  return (
    <Section id="auditor">
      <p className="text-[13px] font-semibold uppercase tracking-[0.14em] mb-4" style={{ color: 'var(--accent)' }}>
        Run it
      </p>
      <h2 className="font-display text-3xl md:text-4xl font-bold leading-tight mb-3 max-w-[22ch]">
        The auditor.
      </h2>
      <p className="text-base leading-relaxed max-w-[58ch] mb-10" style={{ color: 'var(--text-secondary)' }}>
        {tier === 'browse'
          ? 'The Python server is not running, so the controls below cannot start anything. Start it and reload to enable them.'
          : tier === 'check'
            ? 'No local model was found, so Generate and Verify will run in their fast, non-LLM mode.'
            : 'The local server is ready. Deterministic generation and checking are available; language-model assistance is optional.'}
      </p>
      <Scope3Section enabled={tier === 'check' || tier === 'full'} />
    </Section>
  );
}

function Footer() {
  const col = 'text-sm leading-relaxed';
  const link = 'transition-opacity duration-200 hover:opacity-70';
  return (
    <footer className="px-6 py-14" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="max-w-6xl mx-auto grid gap-10 sm:grid-cols-3">
        <div>
          <p className="font-display text-[17px] font-bold mb-3">Scope 3 Auditor</p>
          <p className={col} style={{ color: 'var(--text-faint)' }}>
            An error-detection pipeline for synthetic Scope 3 supply-chain data, and a public record
            of how well it works.
          </p>
        </div>
        <div>
          <p className="text-sm font-bold mb-3">Project</p>
          <ul className={`${col} space-y-2`} style={{ color: 'var(--text-faint)' }}>
            {PUBLIC_DEMO ? <li>Source and instructions are in this downloaded archive.</li> : <>
              <li><a className={link} href={REPO}>View on GitHub</a></li>
              <li><a className={link} href={`${REPO}/blob/main/plan.md`}>Plan and measurements</a></li>
              <li><a className={link} href={`${REPO}/blob/main/architecture.md`}>Architecture</a></li>
            </>}
          </ul>
        </div>
        <div>
          <p className="text-sm font-bold mb-3">Data</p>
          <p className={col} style={{ color: 'var(--text-faint)' }}>
            {PUBLIC_DEMO ? 'This runnable demonstration uses invented teaching factors and waste codes. See PUBLIC-DEMO.md for limits.' : "Emission factors from Thailand's published TGO figures where they exist. Provenance for every row is recorded in the repository."}
          </p>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  const { tier, detail } = useBackend();
  return (
    <>
      <a
        href="#top"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-3 focus:px-4 focus:py-2"
        style={{ background: 'var(--accent)', color: 'var(--on-accent)', borderRadius: 'var(--radius-sm)' }}
      >
        Skip to content
      </a>
      <Nav />
      <main>
        <Hero />
        <TierStrip tier={tier} detail={detail} />
        <Results />
        <HowItWorks />
        <Coverage />
        <Auditor tier={tier} />
      </main>
      <Footer />
    </>
  );
}
