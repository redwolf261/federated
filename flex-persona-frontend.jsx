import { useState, useEffect, useRef } from "react";

const B = {
  dark:      '#141413',
  light:     '#faf9f5',
  midGray:   '#b0aea5',
  lightGray: '#e8e6dc',
  orange:    '#d97757',
  blue:      '#6a9bcc',
  green:     '#788c5d',
  darkCard:  '#1e1e1c',
  darkBorder:'#2a2a28',
};

function injectFonts() {
  if (document.getElementById('flex-fonts')) return;
  const link = document.createElement('link');
  link.id = 'flex-fonts';
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap';
  document.head.appendChild(link);
}

// ── Animated federated-learning network canvas ──────────────────────────────
function NetworkCanvas() {
  const canvasRef = useRef(null);
  const raf = useRef(null);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    const W = cvs.width, H = cvs.height;

    // Spec layout: 3 cluster centroids (left/center-top/right) + 18 clients (6 orbiters each)
    const centroids = [
      { cx: W * 0.22, cy: H * 0.52, color: B.orange }, // left
      { cx: W * 0.50, cy: H * 0.30, color: B.blue },   // center-top
      { cx: W * 0.78, cy: H * 0.52, color: B.green },  // right
    ];

    const server = { x: W * 0.52, y: H * 0.58 };

    // deterministic PRNG so the canvas doesn't "jump"
    let seed = 1337;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };

    const distGate = 110;

    // clients: 6 per centroid orbiting with gentle spring toward orbit radius
    const clients = [];
    const orbitRBase = 60;
    const orbitRVar = 18;

    centroids.forEach((cl, ci) => {
      for (let i = 0; i < 6; i++) {
        const theta = (i / 6) * Math.PI * 2 + rand() * 0.4;
        const r = orbitRBase + rand() * orbitRVar;

        clients.push({
          cluster: ci,
          x: cl.cx + Math.cos(theta) * r,
          y: cl.cy + Math.sin(theta) * r,
          vx: (rand() - 0.5) * 0.1,
          vy: (rand() - 0.5) * 0.1,
          theta,
          r,
          // small angular velocity (tangential motion)
          w: (rand() - 0.5) * 0.035 + (ci - 1) * 0.004,
          pulse: rand() * Math.PI * 2,
          radius: 4.0 + rand() * 2.2,
        });
      }
    });

    // one pulsing packet dot per centroid with staggered timing
    const packet = centroids.map((_, ci) => ({
      offset: ci * 0.17 + rand() * 0.08,
      speed: 0.14 + ci * 0.02,
    }));

    let t = 0;
    const dt = 1 / 60;

    function easeInOutQuad(x) {
      return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      t += 0.95 * dt * 60;

      // update clients (gentle spring + tangential orbit)
      clients.forEach((n) => {
        const cl = centroids[n.cluster];
        n.theta += n.w;

        const targetX = cl.cx + Math.cos(n.theta) * n.r;
        const targetY = cl.cy + Math.sin(n.theta) * n.r;

        const ax = (targetX - n.x) * 0.004;
        const ay = (targetY - n.y) * 0.004;

        n.vx = (n.vx + ax) * 0.94;
        n.vy = (n.vy + ay) * 0.94;

        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.06;
      });

      // intra-cluster edges: thin, opacity based on distance < 110px
      for (let a = 0; a < clients.length; a++) {
        for (let b = a + 1; b < clients.length; b++) {
          if (clients[a].cluster !== clients[b].cluster) continue;

          const d = Math.hypot(clients[a].x - clients[b].x, clients[a].y - clients[b].y);
          if (d > distGate) continue;

          const opacity = Math.max(0, 1 - d / distGate);
          const color = centroids[clients[a].cluster].color;

          ctx.beginPath();
          ctx.moveTo(clients[a].x, clients[a].y);
          ctx.lineTo(clients[b].x, clients[b].y);
          ctx.strokeStyle = color + Math.floor(opacity * 190).toString(16).padStart(2, "0");
          ctx.lineWidth = 0.85;
          ctx.stroke();
        }
      }

      // dashed centroid → server edges + pulsing packets traveling along paths
      centroids.forEach((cl, ci) => {
        // dashed path line
        ctx.beginPath();
        ctx.moveTo(cl.cx, cl.cy);
        ctx.lineTo(server.x, server.y);
        ctx.setLineDash([6, 7]);
        ctx.strokeStyle = cl.color + "30";
        ctx.lineWidth = 1.05;
        ctx.stroke();
        ctx.setLineDash([]);

        // packet position along the path
        const p = packet[ci];
        const progress = (t * p.speed + p.offset) % 1;
        const eased = easeInOutQuad(progress);

        const px = cl.cx + (server.x - cl.cx) * eased;
        const py = cl.cy + (server.y - cl.cy) * eased;

        // packet dot + subtle trail
        ctx.beginPath();
        ctx.arc(px, py, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = cl.color + "cc";
        ctx.fill();

        const trailLen = 12;
        for (let k = 1; k <= trailLen; k++) {
          const q = Math.max(0, progress - k * 0.02);
          const qe = easeInOutQuad(q);
          const tx = cl.cx + (server.x - cl.cx) * qe;
          const ty = cl.cy + (server.y - cl.cy) * qe;

          const a = Math.max(0, 1 - k / trailLen);
          ctx.beginPath();
          ctx.arc(tx, ty, 2.0 * a, 0, Math.PI * 2);
          ctx.fillStyle = cl.color + Math.floor(a * 200).toString(16).padStart(2, "0");
          ctx.fill();
        }

        // centroid rings
        const ringPulse = Math.sin(t * 1.6 + ci) * 0.12 + 0.88;

        ctx.beginPath();
        ctx.arc(cl.cx, cl.cy, 20 * ringPulse, 0, Math.PI * 2);
        ctx.strokeStyle = cl.color + "55";
        ctx.lineWidth = 1.55;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cl.cx, cl.cy, 8.2, 0, Math.PI * 2);
        ctx.fillStyle = cl.color + "b8";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(cl.cx, cl.cy, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = cl.color;
        ctx.fill();
      });

      // client nodes: colored fill + white center dot + subtle pulse ring
      clients.forEach((n) => {
        const cl = centroids[n.cluster];
        const glow = Math.sin(n.pulse) * 0.18 + 0.82;

        // outer glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + 2.6, 0, Math.PI * 2);
        ctx.fillStyle = cl.color + "12";
        ctx.fill();

        // main body
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius * glow, 0, Math.PI * 2);
        ctx.fillStyle = cl.color + "c0";
        ctx.fill();

        // center white dot
        ctx.beginPath();
        ctx.arc(n.x, n.y, Math.max(1.8, n.radius * 0.38), 0, Math.PI * 2);
        ctx.fillStyle = B.light;
        ctx.fill();

        // subtle pulse ring
        const rr = n.radius * (1.05 + Math.sin(t + n.cluster) * 0.06);
        ctx.beginPath();
        ctx.arc(n.x, n.y, rr, 0, Math.PI * 2);
        ctx.strokeStyle = cl.color + "20";
        ctx.lineWidth = 0.9;
        ctx.stroke();
      });

      // server node: white circle with outer glow ring pulsing scale
      const sp = Math.sin(t * 1.25) * 0.18 + 1.0;

      ctx.beginPath();
      ctx.arc(server.x, server.y, 10 * sp, 0, Math.PI * 2);
      ctx.strokeStyle = B.light + "22";
      ctx.lineWidth = 2.0;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(server.x, server.y, 16, 0, Math.PI * 2);
      ctx.fillStyle = B.light + "12";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(server.x, server.y, 9.4, 0, Math.PI * 2);
      ctx.fillStyle = B.light;
      ctx.fill();

      raf.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(raf.current);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={680}
      height={320}
      style={{ width: "100%", height: "auto", opacity: 0.94, display: "block" }}
    />
  );
}

// ── Metric card ─────────────────────────────────────────────────────────────
function MetricCard({ value, label, sub, accent }) {
  const [shown, setShown] = useState(false);
  useEffect(() => { const t = setTimeout(() => setShown(true), 300); return () => clearTimeout(t); }, []);
  return (
    <div style={{
      background: B.darkCard, border: `1px solid ${B.darkBorder}`,
      borderRadius: 12, padding: '20px 22px',
      borderTop: `3px solid ${accent}`,
      transition: 'transform 0.2s',
    }}
      onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-3px)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
    >
      <div style={{ fontFamily: "'Poppins', sans-serif", fontSize: 30, fontWeight: 700, color: accent, lineHeight: 1 }}>
        {shown ? value : '—'}
      </div>
      <div style={{ fontFamily: "'Poppins', sans-serif", fontSize: 13, fontWeight: 600, color: B.light, marginTop: 6, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ fontFamily: "'Lora', Georgia, serif", fontSize: 12, color: B.midGray, marginTop: 4, fontStyle: 'italic' }}>
        {sub}
      </div>
    </div>
  );
}

// ── Feature pill ─────────────────────────────────────────────────────────────
function FeatureCard({ icon, title, desc, accent }) {
  return (
    <div style={{
      background: B.darkCard, border: `1px solid ${B.darkBorder}`,
      borderRadius: 10, padding: '18px 20px', display: 'flex', gap: 14,
    }}>
      <div style={{
        width: 36, height: 36, borderRadius: 8,
        background: accent + '22', color: accent,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 18, flexShrink: 0,
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontFamily: "'Poppins', sans-serif", fontSize: 14, fontWeight: 600, color: B.light, marginBottom: 4 }}>
          {title}
        </div>
        <div style={{ fontFamily: "'Lora', Georgia, serif", fontSize: 13, color: B.midGray, lineHeight: 1.6 }}>
          {desc}
        </div>
      </div>
    </div>
  );
}

// ── Architecture step ────────────────────────────────────────────────────────
function ArchStep({ num, title, detail, accent, last }) {
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <div style={{
          width: 36, height: 36, borderRadius: '50%', background: accent + '22',
          border: `2px solid ${accent}`, color: accent,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: "'Poppins', sans-serif", fontWeight: 700, fontSize: 13,
        }}>{num}</div>
        {!last && <div style={{ width: 2, height: 32, background: accent + '30', marginTop: 4 }} />}
      </div>
      <div style={{ paddingTop: 6 }}>
        <div style={{ fontFamily: "'Poppins', sans-serif", fontSize: 14, fontWeight: 600, color: B.light, marginBottom: 4 }}>
          {title}
        </div>
        <div style={{ fontFamily: "'Lora', Georgia, serif", fontSize: 13, color: B.midGray, lineHeight: 1.65 }}>
          {detail}
        </div>
      </div>
    </div>
  );
}

// ── Badge ────────────────────────────────────────────────────────────────────
function Badge({ label, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '3px 10px', borderRadius: 20,
      background: color + '20', color: color, border: `1px solid ${color}45`,
      fontFamily: "'Poppins', sans-serif", fontSize: 11, fontWeight: 500,
      letterSpacing: '0.03em',
    }}>
      {label}
    </span>
  );
}

// ── Tag chip ─────────────────────────────────────────────────────────────────
function DatasetTag({ label }) {
  return (
    <span style={{
      display: 'inline-block', padding: '4px 12px', borderRadius: 6,
      background: B.darkBorder, color: B.midGray,
      fontFamily: "'Poppins', sans-serif", fontSize: 11, fontWeight: 500,
      border: `1px solid ${B.darkBorder}`,
      letterSpacing: '0.02em',
    }}>
      {label}
    </span>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function FlexPersona() {
  useEffect(() => { injectFonts(); }, []);

  const metrics = [
    { value: '↑ 12%', label: 'Worst-Client Acc.', sub: '', accent: B.orange },
    { value: '↑ 8%',  label: 'Personalization Gain', sub: '', accent: B.blue   },
    { value: '↓ 40%', label: 'Comm. Overhead', sub: '', accent: B.green  },
    { value: '3×',    label: 'Faster Convergence', sub: '', accent: B.orange },
  ];

  const features = [
    { icon: '⚙', title: 'Local Client Training',       accent: B.orange, desc: 'Clients train heterogeneous architectures locally — CNNs, MobileNets, custom backbones — without any weight sharing.' },
    { icon: '🔗', title: 'Representation Adapter',      accent: B.blue,   desc: 'A lightweight projection head maps each client\'s penultimate layer to a shared fixed-dim latent space.' },
    { icon: '🔒', title: 'Secure Representation Sharing', accent: B.green, desc: 'Only compact embedding vectors are exchanged. Raw data and model weights never leave the client.' },
    { icon: '🔁', title: 'Dynamic Client Clustering',   accent: B.orange, desc: 'Cosine similarity over shared representations drives adaptive k-means clustering each round.' },
    { icon: '📡', title: 'Cluster-Level Aggregation',   accent: B.blue,   desc: 'Weighted averaging within clusters merges compatible knowledge before broadcast to cluster members.' },
    { icon: '🎯', title: 'Personalized Head Fine-Tuning', accent: B.green, desc: 'Each client fine-tunes its classifier head on local data post-aggregation for task-specific adaptation.' },
  ];

  const archSteps = [
    { num: '01', title: 'Local train + encode',         accent: B.orange, detail: 'Client trains locally, then passes a batch through the frozen backbone to extract a compact representation vector via the adapter head.' },
    { num: '02', title: 'Secure representation upload', accent: B.blue,   detail: 'Compressed embedding (d=128) is encrypted and uploaded to the FL server — ≈99% smaller than full weights.' },
    { num: '03', title: 'Similarity-driven clustering', accent: B.green,  detail: 'Server computes pairwise cosine similarity, runs adaptive k-means, and assigns each client to the most compatible cluster.' },
    { num: '04', title: 'Cluster-aware aggregation',    accent: B.orange, detail: 'Knowledge representations are aggregated within each cluster. Clients download only their cluster\'s aggregated signal.' },
    { num: '05', title: 'Personalised fine-tuning',     accent: B.blue,   detail: 'Clients integrate cluster knowledge and fine-tune their local head for one or two local epochs before the next round.', last: true },
  ];

  return (
    <div style={{ background: B.dark, minHeight: '100vh', color: B.light, fontFamily: "'Poppins', sans-serif" }}>

      {/* ── Nav ── */}
      <nav style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 32px', borderBottom: `1px solid ${B.darkBorder}`,
        position: 'sticky', top: 0, zIndex: 10, background: B.dark + 'f2',
        backdropFilter: 'blur(8px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: B.orange, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 700, color: '#fff' }}>F</div>
          <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-0.01em' }}>FLEX-Persona</span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['Overview', 'Architecture', 'Evaluation'].map(l => (
            <span key={l} style={{ fontSize: 12, color: B.midGray, padding: '5px 12px', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>{l}</span>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Badge label="Research Preview" color={B.orange} />
        </div>
      </nav>

      {/* ── Hero ── */}
      <section style={{ padding: '60px 32px 40px', maxWidth: 960, margin: '0 auto' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 48, alignItems: 'center' }}>
          <div style={{ flex: '1 1 340px', minWidth: 300 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
              <Badge label="Federated Learning" color={B.orange} />
              <Badge label="Non-IID"            color={B.blue} />
              <Badge label="Personalization"    color={B.green} />
            </div>
            <h1 style={{
              fontFamily: "'Poppins', sans-serif", fontSize: 34, fontWeight: 700,
              lineHeight: 1.12, letterSpacing: '-0.02em', margin: '0 0 16px',
              color: B.light,
            }}>
              FLEX-Persona: Cross-Architecture<br />
              <span style={{ color: B.orange }}>Personalized</span> Federated Learning for Non-IID and<br />
              Heterogeneous Clients
            </h1>
            <p style={{
              fontFamily: "'Lora', Georgia, serif", fontSize: 15, color: B.midGray,
              lineHeight: 1.75, margin: '0 0 28px', maxWidth: 440,
            }}>
              FLEX-Persona enables privacy-preserving collaboration across heterogeneous client models through lightweight representation sharing, dynamic clustering, and cluster-aware personalization — no weight sharing required.
            </p>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <button style={{
                background: B.orange, color: '#fff', border: 'none',
                borderRadius: 8, padding: '10px 22px', fontFamily: "'Poppins', sans-serif",
                fontWeight: 600, fontSize: 13, cursor: 'pointer', letterSpacing: '0.02em',
              }}>
                View Paper →
              </button>
              <button style={{
                background: 'transparent', color: B.light, border: `1px solid ${B.darkBorder}`,
                borderRadius: 8, padding: '10px 22px', fontFamily: "'Poppins', sans-serif",
                fontWeight: 500, fontSize: 13, cursor: 'pointer',
              }}>
                GitHub Repo
              </button>
            </div>
          </div>

          <div style={{ flex: '1 1 320px', minWidth: 280 }}>
            <div style={{
              background: B.darkCard, border: `1px solid ${B.darkBorder}`,
              borderRadius: 16, overflow: 'hidden',
            }}>
              <div style={{ padding: '12px 16px', borderBottom: `1px solid ${B.darkBorder}`, display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: B.orange }} />
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: B.blue }} />
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: B.green }} />
                <span style={{ fontSize: 11, color: B.midGray, marginLeft: 4, fontFamily: 'monospace' }}>live — 3 clusters · 18 clients</span>
              </div>
              <NetworkCanvas />
              <div style={{ padding: '10px 16px', borderTop: `1px solid ${B.darkBorder}`, display: 'flex', justifyContent: 'space-between' }}>
                {[['Cluster A', B.orange], ['Cluster B', B.blue], ['Cluster C', B.green]].map(([l, c]) => (
                  <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <div style={{ width: 7, height: 7, borderRadius: '50%', background: c }} />
                    <span style={{ fontSize: 11, color: B.midGray, fontWeight: 500 }}>{l}</span>
                  </div>
                ))}
                <span style={{ fontSize: 11, color: B.midGray }}>● Server</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Metrics ── */}
      <section style={{ padding: '0 32px 56px', maxWidth: 960, margin: '0 auto' }}>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 14,
        }}>
          {metrics.map(m => <MetricCard key={m.label} {...m} />)}
        </div>
      </section>

      {/* ── Architecture ── */}
      <section style={{ padding: '0 32px 56px', maxWidth: 960, margin: '0 auto' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 40 }}>
          <div style={{ flex: '0 0 auto' }}>
            <div style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: B.orange, fontWeight: 600, marginBottom: 8 }}>
              System Flow
            </div>
            <h2 style={{ fontFamily: "'Poppins', sans-serif", fontSize: 24, fontWeight: 700, margin: '0 0 28px', letterSpacing: '-0.01em' }}>
              How FLEX-Persona<br />works
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {archSteps.map(s => <ArchStep key={s.num} {...s} />)}
            </div>
          </div>

          <div style={{ flex: '1 1 300px', minWidth: 260 }}>
            <div style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: B.blue, fontWeight: 600, marginBottom: 8 }}>
              Tech Stack
            </div>
            <h2 style={{ fontFamily: "'Poppins', sans-serif", fontSize: 24, fontWeight: 700, margin: '0 0 20px', letterSpacing: '-0.01em' }}>
              Implementation
            </h2>
            {[
              { label: 'Language',        val: 'Python 3.10+',                   accent: B.orange },
              { label: 'DL Framework',    val: 'PyTorch 2.x',                    accent: B.blue   },
              { label: 'FL Utilities',    val: 'Flower (flwr)',                   accent: B.green  },
              { label: 'Clustering',      val: 'scikit-learn k-means',            accent: B.orange },
              { label: 'Experiment Mgmt', val: 'Weights & Biases / MLflow',       accent: B.blue   },
              { label: 'Datasets',        val: 'FEMNIST · CIFAR-100',             accent: B.green  },
            ].map(row => (
              <div key={row.label} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '11px 0', borderBottom: `1px solid ${B.darkBorder}`,
              }}>
                <span style={{ fontSize: 12, color: B.midGray, fontWeight: 500 }}>{row.label}</span>
                <span style={{ fontSize: 12, color: row.accent, fontWeight: 600, fontFamily: 'monospace' }}>{row.val}</span>
              </div>
            ))}

            <div style={{ marginTop: 24, padding: '16px 18px', background: B.darkCard, border: `1px solid ${B.darkBorder}`, borderRadius: 10, borderLeft: `3px solid ${B.orange}` }}>
              <div style={{ fontSize: 11, color: B.orange, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>Privacy Guarantee</div>
              <p style={{ fontFamily: "'Lora', Georgia, serif", fontSize: 13, color: B.midGray, margin: 0, lineHeight: 1.65, fontStyle: 'italic' }}>
                Only 128-dim representation vectors leave the client. Raw data and model weights remain local, satisfying communication-efficient and data-minimization privacy properties.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section style={{ padding: '0 32px 56px', maxWidth: 960, margin: '0 auto' }}>
        <div style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: B.green, fontWeight: 600, marginBottom: 8 }}>
          System Modules
        </div>
        <h2 style={{ fontFamily: "'Poppins', sans-serif", fontSize: 24, fontWeight: 700, margin: '0 0 24px', letterSpacing: '-0.01em' }}>
          Key components
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
          {features.map(f => <FeatureCard key={f.title} {...f} />)}
        </div>
      </section>

      {/* ── Evaluation ── */}
      <section style={{ padding: '0 32px 56px', maxWidth: 960, margin: '0 auto' }}>
        <div style={{ background: B.darkCard, border: `1px solid ${B.darkBorder}`, borderRadius: 14, padding: '28px 32px' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 32, justifyContent: 'space-between' }}>
            <div style={{ flex: '1 1 260px' }}>
              <div style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: B.blue, fontWeight: 600, marginBottom: 10 }}>Evaluation Setup</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[
                  ['Datasets',       'FEMNIST (non-IID letter split) · CIFAR-100 (Dirichlet α=0.1, 0.5)'],
                  ['Architectures',  'MobileNetV2, ResNet-18, ShuffleNet, custom lightweight CNNs'],
                  ['Clients',        '50–200 participants, 10–30% participation rate per round'],
                  ['Baselines',      'FedAvg · FedProx · IFCA · pFedMe · Ditto'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                    <span style={{ fontSize: 11, color: B.midGray, fontWeight: 600, flexShrink: 0, minWidth: 80 }}>{k}</span>
                    <span style={{ fontFamily: "'Lora', Georgia, serif", fontSize: 12, color: B.light + 'bb', lineHeight: 1.55 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ flex: '1 1 240px' }}>
              <div style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: B.blue, fontWeight: 600, marginBottom: 10 }}>Evaluation Metrics</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  ['Mean Accuracy',         B.light],
                  ['Worst-Client Accuracy',  B.orange],
                  ['Worst-Group Accuracy',   B.orange],
                  ['Personalization Gain',   B.blue],
                  ['Communication Overhead', B.green],
                  ['Convergence Rounds',     B.green],
                ].map(([m, c]) => (
                  <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: c, flexShrink: 0 }} />
                    <span style={{ fontFamily: "'Poppins', sans-serif", fontSize: 12, color: B.midGray, fontWeight: 500 }}>{m}</span>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 20, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <DatasetTag label="Non-IID" />
                <DatasetTag label="Dirichlet partition" />
                <DatasetTag label="Heterogeneous" />
                <DatasetTag label="Fairness-aware" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{ padding: '20px 32px 32px', borderTop: `1px solid ${B.darkBorder}`, maxWidth: 960, margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 20, height: 20, borderRadius: 4, background: B.orange, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color: '#fff' }}>F</div>
            <span style={{ fontFamily: "'Poppins', sans-serif", fontSize: 12, color: B.midGray, fontWeight: 500 }}>FLEX-Persona · Research Project 2025</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Badge label="PyTorch"  color={B.orange} />
            <Badge label="Flower"   color={B.blue}   />
            <Badge label="FL · PFL" color={B.green}  />
          </div>
        </div>
      </footer>

    </div>
  );
}
