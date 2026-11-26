/**
 * Server-side Stage 1 sampler for streamed chart tiles.
 *
 * Regenerates the tectonic planet from its seed (same code the globe page runs) and prints the
 * conditioning / ghost rasters for the requested chart tiles, so the Stage 2 server can
 * synthesize neighbours without a browser tab doing the sampling.
 *
 * stdin : { seed, archetype, climateMode, seaLevelBias, reliefScale,
 *           homeLatDeg, homeLonDeg, spanDeg,
 *           tiles: [{ qi, qj, coarseSize?, margin?, ghostSize? }] }
 *           A tile with coarseSize gets conditioning; one with ghostSize gets ghost elevation.
 * stdout: { tiles: [{ qi, qj, latDeg, lonDeg, ...conditioning?, ghostElev?, ghostSize? }] }
 */
import {
  chartTileLatLon,
  encodeChartTilePayload,
  extractChartTile,
  extractGhostElev,
  float32ToBase64,
  generateStage1Planet,
} from '../viewer/src/stage1Conditioning.js';

async function readStdin() {
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  return Buffer.concat(chunks).toString('utf8');
}

const req = JSON.parse(await readStdin());
const reliefScale = Number(req.reliefScale ?? 1.0);
const chart = {
  reliefScale,
  homeLatDeg: Number(req.homeLatDeg),
  homeLonDeg: Number(req.homeLonDeg),
  spanDeg: Number(req.spanDeg ?? 46),
};
const planet = generateStage1Planet({
  seed: Number(req.seed),
  archetype: req.archetype || 'earth',
  climateMode: req.climateMode || 'earth',
  seaLevelBias: Number(req.seaLevelBias ?? 0),
});

const tiles = (req.tiles || []).map((t) => {
  const qi = Number(t.qi);
  const qj = Number(t.qj);
  const ll = chartTileLatLon(chart.homeLatDeg, chart.homeLonDeg, chart.spanDeg, qi, qj);
  const out = { qi, qj, latDeg: Number(ll.latDeg.toFixed(4)), lonDeg: Number(ll.lonDeg.toFixed(4)) };
  if (t.coarseSize) {
    Object.assign(out, encodeChartTilePayload(extractChartTile(planet, { ...chart, qi, qj, coarseSize: Number(t.coarseSize), margin: Boolean(t.margin) })));
  }
  if (t.ghostSize) {
    out.ghostSize = Number(t.ghostSize);
    out.ghostElev = float32ToBase64(extractGhostElev(planet, { ...chart, qi, qj, size: out.ghostSize }));
  }
  return out;
});

process.stdout.write(JSON.stringify({ tiles }));
