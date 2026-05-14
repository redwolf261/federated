# FLEX-Persona Research Frontend - Build Plan

## 📋 Project Overview
**Goal**: Single-page research project frontend showcasing FLEX-Persona federated learning framework
**Format**: Single HTML file with embedded React (or pure HTML/CSS/JS for minimal dependencies)
**Theme**: Dark mode with Anthropic brand colors
**Key Challenge**: Canvas-based network visualization animation

---

## 🎨 Design System

### Color Palette (Strict Brand Compliance)
```
Background:      #141413 (primary dark)
Surface/Cards:   #1e1e1c (card containers)
Borders:         #2a2a28 (dividers, subtle outlines)
Text Primary:    #faf9f5 (body, headings)
Text Muted:      #b0aea5 (secondary, captions)

Accent Orange:   #d97757 (CTAs, worst-client metrics, key steps)
Accent Blue:     #6a9bcc (data/network elements)
Accent Green:    #788c5d (efficiency, privacy indicators)
```

### Typography
- **Headings/UI Elements**: Poppins (import Google Fonts)
- **Body/Descriptions**: Lora serif
- **Callouts/Quotes**: Lora serif italic

---

## 📐 Layout Structure

### 1. Navigation (Sticky Header)
- Logo + "FLEX-Persona" text
- Nav links: [Overview, Architecture, Features, Evaluation, GitHub]
- "Research Preview" badge (top-right)
- Backdrop blur, semi-transparent background

### 2. Hero Section
- Large headline with **one accent word** in orange
- Description in Lora (multi-line, ~100 words)
- Two CTA buttons: "View Paper" (orange fill), "Get Started" (orange outline)
- Canvas animation: 18 client nodes (3 clusters × 6 nodes) + 1 server node
  - Position: right side of hero, 600×600px canvas

### 3. Metrics Row (4 Cards)
Cards display:
1. **Worst-Client Accuracy**: ↑12% (orange top border)
2. **Personalization Gain**: ↑8% (blue top border)
3. **Communication Overhead**: ↓40% (green top border)
4. **Convergence Speed**: 3× faster (orange top border)

Each card:
- Top 3px colored border
- Dark background with hover effect
- Large metric number, small label

### 4. Architecture Section
**A. Process Flow (5-Step Numbered)**
1. Local Training → Optimize local loss
2. Upload Representations → Adapter outputs
3. Clustering → Wasserstein distance grouping
4. Aggregation → Cluster prototypes
5. Fine-tuning → Personalized head

**B. Tech Stack Table**
Columns: Component | Technologies
- Model Training: PyTorch, torchvision
- Federated Framework: Flower
- Data Partitioning: scikit-learn
- Optimization: scipy
- Experiment Tracking: Weights & Biases
- Datasets: FEMNIST, CIFAR-100

**C. Privacy Callout Box**
Boxed quote in Lora italic with green left border:
*"Clients only share compact representation summaries, never raw data or full model parameters—ensuring differential privacy guarantees."*

### 5. Features Grid (6 Cards)
2×3 grid of feature cards:
1. **Local Client Training** — Each client optimizes independently
2. **Representation Adapter** — Neural networks map to shared latent space
3. **Secure Representation Sharing** — Compact summaries reduce bandwidth
4. **Dynamic Clustering** — Server groups similar clients automatically
5. **Cluster-Level Aggregation** — Shared knowledge from cluster peers
6. **Personalized Head Fine-Tuning** — Individual task layers per client

Each card: Icon-like visual, title, 1-line description

### 6. Evaluation Panel
**A. Datasets**
- FEMNIST (3597 writers, character recognition)
- CIFAR-100 (Dirichlet α=0.1, α=0.5)

**B. Client Architectures**
- SmallCNN (~100K params)
- ResNet8 (8-layer lightweight)
- MLP (feedforward)

**C. Baseline Comparisons**
- FedAvg
- FedProx
- IFCA
- pFedMe
- Ditto

**D. Metrics with Colored Indicators**
- Accuracy (orange dot)
- Privacy (green dot)
- Communication (blue dot)
- Convergence Time (orange dot)
- Personalization (green dot)

### 7. Footer
- Logo + "FLEX-Persona"
- Tech badges (PyTorch, Flower, FEMNIST, CIFAR-100)
- Links (GitHub, Paper, Authors)
- Copyright text

---

## 🎬 Canvas Animation Specification

### Scene Setup
- Canvas: 600×600px, positioned in hero right side
- Background: transparent (overlays hero gradient)
- Frame rate: 60fps with requestAnimationFrame

### Node Positions
**Cluster Centroids** (fixed):
- Cluster 1 (Blue): (150, 200)
- Cluster 2 (Orange): (300, 400)
- Cluster 3 (Green): (450, 200)
- Server (Center): (300, 300)

**Client Nodes** (6 per cluster, orbit centroids):
- Start angles: 0°, 60°, 120°, 180°, 240°, 300°
- Orbit radius: ~80px with gentle spring physics
- Wobble amplitude: ±15px

### Visual Elements

**Nodes**:
- Client nodes: 8px radius, colored fill (cluster color), white 2px center dot
- Server node: 14px radius, white fill, outer glow ring (pulse scale 1.0→1.3)
- Subtle pulse animation: 2-second cycle

**Edges**:
- Intra-cluster edges: thin lines (1px), cluster color, opacity based on distance
  - Visible only if distance < 110px
  - Opacity = (110 - distance) / 110 × 0.5
- Centroid→Server dashes: 2px dashed lines, cluster color, 0.3 opacity

**Packets**:
- Small circles (4px), travel centroid→server every 1.5s
- Staggered: each cluster offset by 0.5s
- Animation: bezier path with ease-out timing
- Color: cluster color with 0.8 opacity

### Physics & Timing
- Spring constant: gentle wobble (~1.5 damping factor)
- Packet speed: variable, 1.2s travel time
- Pulse: sin-wave, 2-second cycle
- Frame skipping: no (60fps constant)

---

## 🛠️ Implementation Approach

### File Structure
```
flex-persona-research-frontend.html (single file, all-in-one)
├── HTML structure (sections, semantic markup)
├── <style> tag (all CSS, CSS variables for colors)
├── <canvas> element (animation)
└── <script> tag (vanilla JS or React CDN)
```

### Technology Stack
- **HTML5**: Semantic markup
- **CSS3**: Grid/Flexbox, custom properties, animations
- **Vanilla JS**: Canvas rendering, animation loop, spring physics
- **Google Fonts**: Poppins, Lora (async load)
- **No external UI libraries** (components built from scratch)

### Build Sequence
1. **Setup**: HTML skeleton, CSS variables, Google Fonts import
2. **Navigation**: Sticky header with nav bar
3. **Hero Section**: Layout + gradient background + CTA buttons
4. **Canvas Animation**: Spring physics engine + render loop
5. **Metrics Cards**: Grid + styling with colored top borders
6. **Architecture Section**: Process flow + tech table + callout box
7. **Features Grid**: 6-card layout with hover effects
8. **Evaluation Panel**: Datasets, architectures, baselines, metrics
9. **Footer**: Links + tech badges
10. **Polish**: Animations, transitions, responsive tweaks

---

## 🎯 Success Criteria

- [ ] Single HTML file, <150KB gzipped
- [ ] All brand colors strictly adhered to
- [ ] Canvas animation smooth (60fps), no jank
- [ ] Fully responsive (mobile/tablet/desktop)
- [ ] Fast load time (<2s on fast connection)
- [ ] Accessibility: semantic HTML, color contrast ≥4.5:1
- [ ] No console errors or warnings
- [ ] All external fonts cached / Google Fonts optimized
- [ ] Print-friendly (footer visible)

---

## 📝 Next Steps

1. **Approve plan** (any adjustments to layout, colors, or animation spec)
2. **Create HTML file** with base structure
3. **Implement canvas animation** (physics + rendering)
4. **Build sections** in order (nav → hero → metrics → ... → footer)
5. **Polish & optimize** (performance, micro-interactions, accessibility)
6. **Test & deploy** (cross-browser, mobile, accessibility audit)

---

## 🔗 References

- **Anthropic Brand**: Colors as specified, Poppins + Lora
- **FLEX-Persona Research**: 18 clients (3×6), clustering via Wasserstein distance
- **Canvas API**: Spring physics, packet animation, requestAnimationFrame
- **Performance**: Minimize reflows, use CSS transforms for animations

