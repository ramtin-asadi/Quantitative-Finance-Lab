// Run from any directory with: node config/website/export-readme-art.mjs
// Reuse the homepage's deterministic SVG geometry; no browser or build is needed.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const source = await readFile(new URL("homepage/app.js", import.meta.url), "utf8");
const start = source.indexOf("const svg =");
const end = source.indexOf("\nfunction drawSurface(");
if (start < 0 || end < start) throw new Error("Homepage graphic functions not found");
const graphics = runInNewContext(
  `${source.slice(start, end)}\n({ marketPathsGraphic })`,
);
const directory = new URL("readme/", import.meta.url);
await mkdir(directory, { recursive: true });

// Keep the same curves in both themes, with strokes readable on dark surfaces.
const darkColors = new Map([
  ["#3973d3", "#8cb4f8"], ["#88a9df", "#b0cafa"],
  ["#d9755c", "#f0a28c"], ["#c3972c", "#e8c979"],
  ["#7196d4", "#9abcf1"], ["#d4dfed", "#3a4a63"],
]);
const hero = graphics.marketPathsGraphic().replace(
  /viewBox="0 0 800 300"/,
  'viewBox="0 0 1000 300"',
).replace("<g ", '<g transform="translate(100 0)" ');

async function save(name, markup, title) {
  const accessible = markup
    .replace('aria-hidden="true"', 'role="img" aria-labelledby="title"')
    .replace(/\sclass="[^"]*"/g, "")
    .replace(/\sstyle="[^"]*"/g, "")
    .replace(/(<svg\b[^>]*>)/, `$1<title id="title">${title}</title>`);
  await writeFile(new URL(`${name}.svg`, directory), accessible + "\n");
  const dark = accessible.replace(
    /#[0-9a-f]{6}\b|\bwhite\b/g,
    (color) => darkColors.get(color) || color,
  );
  await writeFile(new URL(`${name}-dark.svg`, directory), dark + "\n");
}

await save("market-paths", hero, "Illustrative market paths and possible outcomes");
console.log(`Exported the README banner to ${fileURLToPath(directory)}`);
