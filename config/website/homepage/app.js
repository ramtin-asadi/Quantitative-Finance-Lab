"use strict";

// On the published homepage, the original Markdown project list is the source.
// The standalone workspace preview provides the same content through projects.js.
const PROJECTS =
  window.LAB_PROJECTS ||
  [...document.querySelectorAll("#project-directory .project-seed")].map(
    (entry) => ({
      id: Number(entry.dataset.projectId),
      anchor: entry.id,
      title: entry
        .querySelector("h3")
        .textContent.replace(/^\s*\d+\.\s*/, "")
        .trim(),
      bullets: [...entry.querySelectorAll("li")].map((item) =>
        item.textContent.trim(),
      ),
      url: entry.querySelector("a").getAttribute("href"),
    }),
  );
document.body.classList.add("js-enabled");
const TRACKS = [
  {
    id: "rates",
    number: "01",
    name: "Rates & credit",
    card: "Rates &<br>credit",
    subtitle: "Yield curves, term structure and credit risk.",
    color: "#3973d3",
    mapColor: "#3973d3",
    ids: [1, 9, 22],
    intro:
      "From the shape of the yield curve to the risk inside a corporate balance sheet.",
    label: "Rates / credit",
  },
  {
    id: "derivatives",
    number: "02",
    name: "Derivatives & volatility",
    card: "Derivatives &<br>volatility",
    subtitle: "Pricing, hedging and numerical methods.",
    color: "#d9755c",
    mapColor: "#d9755c",
    ids: [4, 5, 7, 8, 11, 13, 14, 18],
    intro:
      "Follow uncertainty through pricing, hedging, volatility surfaces and numerical methods.",
    label: "Derivatives / vol",
  },
  {
    id: "portfolio",
    number: "03",
    name: "Portfolios & risk",
    card: "Portfolios &<br>risk",
    subtitle: "Allocation, dependence and risk.",
    color: "#b3892b",
    mapColor: "#b3892b",
    ids: [2, 3, 6, 10, 15, 16, 17],
    intro:
      "Turn estimates, views, regimes and dependencies into portfolio decisions you can evaluate.",
    label: "Portfolios / risk",
  },
  {
    id: "macro",
    number: "04",
    name: "Macro & fundamentals",
    card: "Macro &<br>fundamentals",
    subtitle: "Economic information and company research.",
    color: "#6c829f",
    mapColor: "#6c829f",
    ids: [12, 21, 23],
    intro:
      "Connect economic conditions, company fundamentals and the information available at the time.",
    label: "Macro / fundamentals",
  },
  {
    id: "intelligence",
    number: "05",
    name: "Machine learning & AI",
    card: "Machine learning<br>& AI",
    subtitle: "Forecasts, learned policies and financial analysis.",
    color: "#348d95",
    mapColor: "#348d95",
    ids: [19, 20, 24],
    intro:
      "Move from probabilistic forecasts to learned allocation policies and a local financial analyst.",
    label: "Machine learning / AI",
  },
];
const SHORT = {
  1: "Yield curves",
  2: "Mean–variance",
  3: "Risk & CAPM",
  4: "Black–Scholes",
  5: "GARCH & VRP",
  6: "Black–Litterman",
  7: "Dynamic hedging",
  8: "Volatility surfaces",
  9: "Term structure",
  10: "Robust portfolios",
  11: "Stochastic vol",
  12: "Financial conditions",
  13: "American options",
  14: "Fourier pricing",
  15: "Factor investing",
  16: "Market regimes",
  17: "Market networks",
  18: "Rough volatility",
  19: "Forecasting & Kelly",
  20: "Reinforcement learning",
  21: "Fundamental research",
  22: "Corporate credit",
  23: "Macro nowcasting",
  24: "The financial analyst",
};
const MAP_NAMES = {
  1: ["Yield", "curves"],
  2: ["Mean–", "variance"],
  3: ["Risk &", "CAPM"],
  4: ["Black–", "Scholes"],
  5: ["GARCH", "& VRP"],
  6: ["Black–", "Litterman"],
  7: ["Dynamic", "hedging"],
  8: ["Volatility", "surfaces"],
  9: ["Term", "structure"],
  10: ["Robust", "portfolios"],
  11: ["Stochastic", "volatility"],
  12: ["Financial", "conditions"],
  13: ["American", "options"],
  14: ["Fourier", "pricing"],
  15: ["Factor", "investing"],
  16: ["Market", "regimes"],
  17: ["Market", "networks"],
  18: ["Rough", "volatility"],
  19: ["Forecasting", "& Kelly"],
  20: ["Reinforcement", "learning"],
  21: ["Fundamental", "research"],
  22: ["Corporate", "credit"],
  23: ["Macro", "nowcasting"],
  24: ["Financial", "analyst"],
};
const projectById = new Map(PROJECTS.map((p) => [p.id, p]));
const trackByProject = new Map(
  TRACKS.flatMap((t) => t.ids.map((id) => [id, t])),
);
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const num = (id) => String(id).padStart(2, "0");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const svg = (body, viewBox = "0 0 320 160", label = "") =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" fill="none" ${label ? `role="img" aria-label="${esc(label)}"` : 'aria-hidden="true"'}>${body}</svg>`;
const pathFrom = (points) =>
  points
    .map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ");

function surfaceHeight(u, v) {
  return (
    0.78 * Math.sqrt((u + 0.28) ** 2 + 0.15) -
    0.23 * u +
    0.16 * Math.sin(v * 1.5) +
    0.07 * v * v
  );
}

function marketPathsGraphic() {
  let state = 190711;
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
  const normal = () =>
    Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
  const steps = 120;
  const paths = Array.from({ length: 56 }, () => {
    let value = 0;
    const samples = [0];
    for (let i = 1; i <= steps; i++) {
      value += normal() / Math.sqrt(steps);
      samples.push(value);
    }
    return samples.map((value, i) =>
      i > 0 && i < steps
        ? 0.2 * samples[i - 1] + 0.6 * value + 0.2 * samples[i + 1]
        : value,
    );
  });
  const extent = Math.max(...paths.flat().map(Math.abs));
  const scale = 126 / extent;
  const point = (value, i) => [58 + (660 * i) / steps, 150 - value * scale];
  const quantile = (i, q) => {
    const values = paths.map((p) => p[i]).sort((a, b) => a - b);
    return values[Math.floor((values.length - 1) * q)];
  };
  const envelope = (low, high) => {
    const upper = Array.from({ length: steps + 1 }, (_, i) =>
      point(quantile(i, high), i),
    );
    const lower = Array.from({ length: steps + 1 }, (_, i) =>
      point(quantile(i, low), i),
    ).reverse();
    return pathFrom([...upper, ...lower]) + " Z";
  };
  let body = `<path d="${envelope(0.03, 0.97)}" fill="#3973d3" opacity=".025"/><path d="${envelope(0.15, 0.85)}" fill="#3973d3" opacity=".035"/>`;
  paths.forEach((values, i) => {
    body += `<path d="${pathFrom(values.map(point))}" stroke="${i % 5 === 0 ? "#88a9df" : "#3973d3"}" stroke-width=".72" opacity="${i % 5 === 0 ? ".38" : ".25"}"/>`;
  });
  const ordered = [...paths].sort((a, b) => a[steps] - b[steps]);
  [0.16, 0.5, 0.84].forEach((q, i) => {
    const values = ordered[Math.floor(q * (ordered.length - 1))];
    const color = ["#d9755c", "#3973d3", "#c3972c"][i];
    const end = point(values[steps], steps);
    body += `<path class="fan-feature" pathLength="1" d="${pathFrom(values.map(point))}" stroke="${color}" stroke-width="${i === 1 ? 1.85 : 1.6}" opacity=".95" style="animation-delay:${i * 0.18}s"/><circle cx="${end[0]}" cy="${end[1]}" r="2" fill="${color}"/>`;
  });
  const endings = paths.map((p) => point(p[steps], steps)[1]);
  const densitySamples = Array.from({ length: 110 }, (_, i) => {
    const y = 19 + (i * 262) / 109;
    const value = endings.reduce(
      (sum, end) => sum + Math.exp(-0.5 * ((y - end) / 15) ** 2),
      0,
    );
    return [y, value];
  });
  const peak = Math.max(...densitySamples.map((p) => p[1]));
  const density = densitySamples.map(([y, value]) => [
    746 + (30 * value) / peak,
    y,
  ]);
  const densityArea = pathFrom(density) + " L746,281 L746,19 Z";
  body += `<path d="${densityArea}" fill="#3973d3" opacity=".055"/><path d="M746 19V281" stroke="#d4dfed" stroke-width=".75"/><path d="${pathFrom(density)}" stroke="#7196d4" stroke-width="1.15"/><circle cx="58" cy="150" r="2.5" fill="#3973d3"/>`;
  return svg(
    `<g fill="none" stroke-linecap="round" stroke-linejoin="round">${body}</g>`,
    "0 0 800 300",
  );
}

function drawSurface() {
  document.getElementById("surface-stage").innerHTML = marketPathsGraphic();
}

function derivativeSurfaceGraphic() {
  const project = (u, v) => [
    400 + u * 205 + v * 113,
    275 + v * 52 - surfaceHeight(u, v) * 215,
  ];
  const steps = 26,
    extent = 1.15;
  let mesh = "";
  for (let i = 0; i <= steps; i++) {
    const fixed = -extent + (i * extent * 2) / steps;
    const row = [],
      col = [];
    for (let j = 0; j <= 65; j++) {
      const moving = -extent + (j * extent * 2) / 65;
      row.push(project(fixed, moving));
      col.push(project(moving, fixed));
    }
    mesh += `<path d="${pathFrom(row)}" stroke="#3973d3" stroke-width="1.05" opacity="${i % 3 === 0 ? 0.9 : 0.55}"/><path d="${pathFrom(col)}" stroke="#9bb7e5" stroke-width=".9" opacity=".68"/>`;
  }
  const highlight = Array.from({ length: 100 }, (_, i) =>
    project(-extent + (i * extent * 2) / 99, 0.22),
  );
  const boundary = Array.from({ length: 100 }, (_, i) =>
    project(-extent + (i * extent * 2) / 99, -extent),
  );
  const section = Array.from({ length: 100 }, (_, i) =>
    project(-0.23, -extent + (i * extent * 2) / 99),
  );
  return svg(
    `<g class="surface-float" stroke-linecap="round" stroke-linejoin="round">${mesh}<path d="${pathFrom(boundary)}" stroke="#3973d3" stroke-width="1.3"/><path d="${pathFrom(section)}" stroke="#c3972c" stroke-width="1.5"/><path class="surface-highlight" d="${pathFrom(highlight)}" stroke="#d9755c" stroke-width="2"/></g>`,
    "0 0 800 275",
  );
}

function trackGraphic(id) {
  if (id === "rates") {
    let paths = "";
    const colors = ["#c2d2ed", "#88a9dd", "#3973d3", "#d99580"];
    for (let line = 0; line < 4; line++) {
      const pts = Array.from({ length: 70 }, (_, i) => {
        const t = i / 69;
        return [
          48 + t * 222,
          105 -
            (1 - Math.exp(-t * (2.4 + line * 0.6))) * (30 + line * 13) +
            Math.sin(t * 5 + line * 0.7) * 4,
        ];
      });
      paths += `<path d="${pathFrom(pts)}" stroke="${colors[line]}" stroke-width="${line === 2 ? 1.8 : 1}"/><circle cx="${pts.at(-1)[0]}" cy="${pts.at(-1)[1]}" r="2.5" fill="${colors[line]}"/>`;
    }
    return svg(
      `<g stroke="#e8ebf1" stroke-width=".6"><path d="M48 30v85h235M48 75h235M48 95h235M102 30v85M158 30v85M214 30v85"/></g>${paths}<path d="M48 115h222" stroke="#d1d9e8" stroke-dasharray="2 4"/>`,
    );
  }
  if (id === "derivatives") {
    return derivativeSurfaceGraphic();
  }
  if (id === "portfolio") {
    const frontier = Array.from({ length: 85 }, (_, i) => {
      const t = i / 84;
      return [70 + 208 * t, 111 - 76 * Math.sqrt(t)];
    });
    const area = pathFrom(frontier) + " L278,126 L70,126 Z";
    let dots = "";
    for (let i = 0; i < 43; i++) {
      const t = ((i * 37 + 19) % 208) / 208;
      const x = 70 + 208 * t;
      const top = 111 - 76 * Math.sqrt(t);
      const y = top + 8 + ((i * 23 + 7) % 39);
      if (y < 122)
        dots += `<circle cx="${x}" cy="${y}" r="${i % 6 === 0 ? 2 : 1.6}" fill="#c3d2e8"/>`;
    }
    const chosen = [0.05, 0.42, 0.88]
      .map((t, i) => {
        const x = 70 + 208 * t,
          y = 111 - 76 * Math.sqrt(t);
        return `<circle cx="${x}" cy="${y}" r="3.3" fill="${["#3973d3", "#b3892b", "#d9755c"][i]}" stroke="white" stroke-width="1.5"/>`;
      })
      .join("");
    return svg(
      `<path d="M48 23V126H285" stroke="#e2e7ef" stroke-width=".8"/><path d="${area}" fill="#f4f7fc"/>${dots}<path d="M50 119L278 39" stroke="#c4a45a" stroke-width=".85" stroke-dasharray="2 4" opacity=".8"/><path d="${pathFrom(frontier)}" stroke="#3973d3" stroke-width="1.8"/>${chosen}<g fill="#8b96a7" font-family="'Lab Sans','Segoe UI',sans-serif" font-size="9"><text x="32" y="20">E[r]</text><text x="284" y="140">σ</text></g>`,
    );
  }
  if (id === "macro") {
    let bars = "";
    for (let i = 0; i < 22; i++) {
      const h = 14 + Math.sin(i * 0.4) * 14 + Math.abs(Math.cos(i * 0.72)) * 25;
      bars += `<rect x="${34 + i * 11.4}" y="${112 - h}" width="5.5" height="${h}" fill="${i % 3 === 0 ? "#e1c574" : "#d9e2f3"}" opacity=".9"/>`;
    }
    const pts = Array.from({ length: 100 }, (_, i) => {
      const t = i / 99;
      return [
        35 + t * 247,
        76 - Math.sin(t * 6.7) * 14 - Math.cos(t * 13) * 6 - t * 25,
      ];
    });
    return svg(
      `<path d="M35 112h247M35 32v80" stroke="#e2e7ee" stroke-width=".8"/>${bars}<path d="${pathFrom(pts)}" stroke="#5279be" stroke-width="1.8"/><circle cx="${pts.at(-1)[0]}" cy="${pts.at(-1)[1]}" r="3" fill="#5279be"/>`,
    );
  }
  const nodes = [];
  for (let layer = 0; layer < 5; layer++)
    for (let i = 0; i < (layer === 2 ? 5 : 3); i++)
      nodes.push({
        layer,
        x: 66 + layer * 48,
        y: layer === 2 ? 33 + i * 19 : 46 + i * 27,
      });
  let body = "";
  nodes.forEach((a) =>
    nodes
      .filter((b) => b.layer === a.layer + 1)
      .forEach(
        (b) =>
          (body += `<path d="M${a.x} ${a.y}C${a.x + 22} ${a.y},${b.x - 22} ${b.y},${b.x} ${b.y}" stroke="#b2c4d4" stroke-width=".8" opacity=".65"/>`),
      ),
  );
  nodes.forEach(
    (n) =>
      (body += `<circle cx="${n.x}" cy="${n.y}" r="${n.layer === 2 ? 3.7 : 2.8}" fill="${n.layer === 2 ? "#348d95" : "#b6cbd7"}" stroke="#ffffff" stroke-width="1.4"/>`),
  );
  return svg(
    `<g class="graphic-pulse">${body}</g><ellipse cx="162" cy="73" rx="45" ry="62" stroke="#bdcbdc" stroke-width=".7" stroke-dasharray="2 5"/>`,
  );
}

function selectTrack(
  id,
  { focus = false, scroll = false, updateHash = false } = {},
) {
  const track = TRACKS.find((t) => t.id === id);
  document.querySelectorAll(".track-card").forEach((button) => {
    const selected = button.dataset.track === id;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  const panel = document.getElementById("track-panel");
  panel.setAttribute("aria-labelledby", `track-${id}`);
  panel.style.setProperty("--selected-color", track.color);
  panel.innerHTML = `<div class="panel-intro"><div class="panel-label">${track.ids.length} projects in this track</div><h3>${esc(track.name)}</h3><p>${esc(track.intro)}</p></div><div class="panel-projects">${track.ids
    .map((id) => {
      const p = projectById.get(id);
      return `<a class="panel-project" href="${esc(p.url)}"><span class="project-num">${num(id)}</span><span>${esc(SHORT[id])}</span><span class="project-link-arrow" aria-hidden="true">↗</span></a>`;
    })
    .join("")}</div>`;
  if (updateHash) history.replaceState(null, "", `#track-${id}`);
  if (scroll)
    panel.scrollIntoView({
      behavior: reducedMotion.matches ? "instant" : "smooth",
      block: "nearest",
    });
}

function renderTracks() {
  const list = document.getElementById("track-cards");
  list.innerHTML = TRACKS.map(
    (t, i) =>
      `<button class="track-card ${t.id}" id="track-${t.id}" data-track="${t.id}" role="tab" aria-controls="track-panel" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" style="--track-color:${t.color}" type="button"><span class="track-meta"><span class="track-number">${t.number}</span><span>${t.ids.length} projects</span></span><span class="track-graphic">${trackGraphic(t.id)}</span><span class="track-name">${t.card}</span><span class="track-subtitle">${esc(t.subtitle)}</span><span class="track-arrow" aria-hidden="true">↗</span></button>`,
  ).join("");
  list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-track]");
    if (button)
      selectTrack(button.dataset.track, { scroll: true, updateHash: true });
  });
  list.addEventListener("keydown", (event) => {
    const keys = [
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "Home",
      "End",
    ];
    if (!keys.includes(event.key)) return;
    const current = TRACKS.findIndex(
      (t) => `track-${t.id}` === document.activeElement.id,
    );
    if (current === -1) return;
    event.preventDefault();
    let next = current;
    if (["ArrowRight", "ArrowDown"].includes(event.key))
      next = (current + 1) % TRACKS.length;
    if (["ArrowLeft", "ArrowUp"].includes(event.key))
      next = (current + TRACKS.length - 1) % TRACKS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TRACKS.length - 1;
    selectTrack(TRACKS[next].id, { focus: true, updateHash: true });
  });
  selectTrack("rates");
}

function renderMap() {
  const positions = new Map();
  TRACKS.forEach((t, lane) =>
    t.ids.forEach((id, i) => {
      const start = t.id === "intelligence" ? 550 : 205;
      const x = start + ((1145 - start) * i) / (t.ids.length - 1),
        y = 65 + lane * 88 + Math.sin(i * 1.3 + lane * 0.6) * 8;
      positions.set(id, [x, y]);
    }),
  );
  const connections = [
    [1, 4],
    [3, 19],
    [23, 24],
  ]
    .map(([a, b]) => {
      const [x1, y1] = positions.get(a),
        [x2, y2] = positions.get(b);
      const offset = x1 === x2 ? -24 : 40;
      return `<path class="map-connection" data-related="${trackByProject.get(a).id} ${trackByProject.get(b).id}" d="M${x1},${y1}C${x1 + offset},${y1 + (y2 - y1) * 0.55} ${x2 + offset},${y2 - (y2 - y1) * 0.55} ${x2},${y2}" stroke="#a7adb9" stroke-width="1" stroke-dasharray="3 5" opacity=".55"/>`;
    })
    .join("");
  let body = `<g font-family="Consolas,monospace" font-size="8" letter-spacing="1.2" fill="#9a9fa9"><text x="205" y="13">Foundations</text><text x="1145" y="13" text-anchor="end">Further investigations →</text></g>${connections}`;
  TRACKS.forEach((t) => {
    const points = t.ids.map((id) => positions.get(id));
    let route = `M${points[0][0]},${points[0][1]}`;
    points.slice(1).forEach((p, i) => {
      const prev = points[i],
        mid = (p[0] + prev[0]) / 2;
      route += `C${mid},${prev[1]} ${mid},${p[1]} ${p[0]},${p[1]}`;
    });
    const laneY = positions.get(t.ids[0])[1];
    body += `<g class="map-lane" data-map-track="${t.id}"><circle cx="3" cy="${laneY - 7}" r="2.8" fill="${t.mapColor}"/><text class="map-track-name" x="14" y="${laneY - 2}" fill="#434b57" font-size="15" font-family="'Lab Serif',Georgia,serif">${esc(t.label)}</text><text class="map-track-count" x="14" y="${laneY + 17}" fill="#737e8e" font-size="10" font-family="'Lab Sans','Segoe UI',sans-serif">${t.ids.length} projects</text><path d="${route}" stroke="${t.mapColor}" stroke-width="1.5" opacity=".75"/>`;
    t.ids.forEach((id) => {
      const [x, y] = positions.get(id),
        p = projectById.get(id),
        lines = MAP_NAMES[id];
      body += `<a class="roadmap-node" href="${esc(p.url)}" aria-label="Project ${num(id)}: ${esc(p.title)}"><title>Project ${num(id)} — ${esc(p.title)}</title><circle class="node-focus" cx="${x}" cy="${y}" r="20" stroke="${t.mapColor}" stroke-width="2" fill="none" opacity="0"/><circle class="node-outer" cx="${x}" cy="${y}" r="13" fill="#ffffff" stroke="${t.mapColor}" stroke-width="1.3"/><text class="node-number" x="${x}" y="${y + 3}" text-anchor="middle" fill="${t.mapColor}" font-size="9.5" font-family="Consolas,monospace">${num(id)}</text><text class="node-title" x="${x}" y="${y + 29}" text-anchor="middle" fill="#687282" font-size="10.5" font-family="'Lab Sans','Segoe UI',sans-serif">${lines.map((line, j) => `<tspan x="${x}" dy="${j ? 12 : 0}">${esc(line)}</tspan>`).join("")}</text></a>`;
    });
    body += "</g>";
  });
  document.getElementById("project-map").innerHTML = svg(
    body,
    "0 0 1190 485",
    "All 24 research projects arranged across five reading tracks, with selected connections between them.",
  );
  document.getElementById("map-legend").innerHTML =
    `<button class="map-filter" type="button" aria-pressed="true" data-highlight="all">Full map</button>${TRACKS.map((t) => `<button class="map-filter" type="button" aria-pressed="false" data-highlight="${t.id}" style="--track-color:${t.mapColor}"><span class="legend-dot" aria-hidden="true"></span>${esc(t.id === "intelligence" ? "ML & AI" : t.name)}</button>`).join("")}`;
  document.getElementById("map-legend").addEventListener("click", (event) => {
    const button = event.target.closest("[data-highlight]");
    if (!button) return;
    const id = button.dataset.highlight;
    document
      .querySelectorAll("[data-highlight]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    document
      .querySelectorAll("[data-map-track]")
      .forEach((lane) =>
        lane.classList.toggle(
          "is-dim",
          id !== "all" && lane.dataset.mapTrack !== id,
        ),
      );
    document
      .querySelectorAll(".map-connection")
      .forEach(
        (path) =>
          (path.style.opacity =
            id === "all" || path.dataset.related.split(" ").includes(id)
              ? ".55"
              : ".08"),
      );
  });
}

function renderDirectory() {
  const search = document
    .getElementById("project-search")
    .value.trim()
    .toLowerCase();
  const filter = document.getElementById("project-filter").value;
  const numberQuery = search.match(/^#?(\d+)$/);
  const matches = PROJECTS.filter((p) => {
    const track = trackByProject.get(p.id);
    if (filter !== "all" && track.id !== filter) return false;
    return (
      !search ||
      (numberQuery
        ? p.id === Number(numberQuery[1])
        : `${p.title} ${p.bullets.join(" ")} ${track.name} ${SHORT[p.id]}`
            .toLowerCase()
            .includes(search))
    );
  });
  document.getElementById("results-count").textContent =
    `${matches.length} ${matches.length === 1 ? "project" : "projects"}`;
  document.getElementById("empty-state").hidden = matches.length > 0;
  document.getElementById("project-directory").innerHTML = matches
    .map((p) => {
      const t = trackByProject.get(p.id);
      return `<article class="project-entry" id="${esc(p.anchor || `project-${num(p.id)}`)}" style="--track-color:${t.color}"><span class="directory-number" aria-hidden="true">${num(p.id)}</span><div class="project-content"><div class="project-tag">${esc(t.name)}</div><h3><a href="${esc(p.url)}"><span class="sr-only">Project ${num(p.id)}: </span>${esc(p.title)}</a></h3><div class="project-controls"><button class="outline-toggle" data-outline="outline-${p.id}" aria-controls="outline-${p.id}" aria-expanded="false" type="button">Project outline <span aria-hidden="true">+</span></button><a class="open-project" href="${esc(p.url)}">Open project <span aria-hidden="true">↗</span></a></div><div class="outline-content" id="outline-${p.id}" hidden><ul>${p.bullets.map((b) => `<li>${esc(b)}</li>`).join("")}</ul></div></div></article>`;
    })
    .join("");
}

function initDirectory() {
  document
    .getElementById("project-filter")
    .insertAdjacentHTML(
      "beforeend",
      TRACKS.map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join(
        "",
      ),
    );
  document
    .getElementById("project-search")
    .addEventListener("input", renderDirectory);
  document
    .getElementById("project-filter")
    .addEventListener("change", renderDirectory);
  document.getElementById("reset-search").addEventListener("click", () => {
    document.getElementById("project-search").value = "";
    document.getElementById("project-filter").value = "all";
    renderDirectory();
    document.getElementById("project-search").focus();
  });
  document
    .getElementById("project-directory")
    .addEventListener("click", (event) => {
      const button = event.target.closest("[data-outline]");
      if (!button) return;
      const open = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!open));
      button.querySelector("span").textContent = open ? "+" : "−";
      document.getElementById(button.dataset.outline).hidden = open;
    });
  renderDirectory();
}

function followPageAnchor() {
  const anchor = location.hash.slice(1);
  const track = TRACKS.find((t) => `track-${t.id}` === anchor);
  if (track) selectTrack(track.id);
  // Track links open the matching tab and keep the section heading in view.
  // Existing section and project anchors keep their original destinations.
  const target = document.getElementById(track ? "tracks" : anchor);
  if (target)
    requestAnimationFrame(() =>
      target.scrollIntoView({ block: "start", behavior: "instant" }),
    );
}

function initHomepage() {
  drawSurface();
  renderTracks();
  renderMap();
  initDirectory();
  followPageAnchor();
  window.addEventListener("hashchange", followPageAnchor);
}

// Quarto first processes the static HTML links. Enhance on the next turn so
// its navigation code never tries to treat SVGAnimatedString as an HTML URL.
if (document.readyState === "complete") {
  initHomepage();
} else {
  document.addEventListener(
    "DOMContentLoaded",
    () => setTimeout(initHomepage, 0),
    {
      once: true,
    },
  );
}
