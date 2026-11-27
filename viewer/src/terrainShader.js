import * as THREE from 'three';

let blankRock = null;
/** A 1×1 rock map of plain sediment, for worlds that have no rock raster. */
export function blankRockTexture() {
  if (!blankRock) {
    blankRock = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    blankRock.needsUpdate = true;
  }
  return blankRock;
}

/**
 * High-Detail Stylized-Realistic GLSL Terrain Shader:
 *   - Real heightmap ray-marched mountain self-shadowing across valleys (soft penumbra)
 *   - Triplanar procedural rock strata, cracks & micro-normal perturbation on cliffs
 *   - Multi-scale ground albedo + micro-bump on meadows/steppe/soil
 *   - Geological Curvature / Cavity AO (gullies darkened, ridges lifted)
 *   - Wet pebble/mud banks beside carved 3D river channels
 *   - Drifting cloud shadows + height-based atmospheric valley mist
 *
 *   Style modes:
 *     0 = High-Detail World (Triplanar Rock + Ray-Marched Shadows + Mist)
 *     1 = Faceted Diorama (Crisp Low-Poly Miniature)
 *     2 = Topographic Atlas (Elevation Contours + Cliff Slope Map + Rivers)
 */
export function createTerrainMaterial(
  maskTex,
  heightFloatTex,
  worldSpan,
  maxElev,
  gridSize,
  climateTex = null,
  hasClimate = false,
  tempRange = [-12.0, 28.0],
  precipRange = [0.0, 1200.0]
) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uMasks: { value: maskTex },
      uHeightTex: { value: heightFloatTex },
      uClimate: { value: climateTex },
      uHasClimate: { value: hasClimate ? 1 : 0 },
      // Bedrock weights (granite, volcanic, limestone, metamorphic) on the climate raster's
      // grid; set by the caller when the world has a rock map
      uRock: { value: blankRockTexture() },
      uHasRock: { value: 0 },
      uTempRange: { value: new THREE.Vector2(tempRange[0], tempRange[1]) },
      uPrecipRange: { value: new THREE.Vector2(precipRange[0], precipRange[1]) },
      uWorldSpan: { value: worldSpan },
      uGridSize: { value: gridSize },
      // 0: node-registered single world (first/last texel on the world edge)
      // 1: cell-registered seam atlas (see seamAtlas.js)
      uCellReg: { value: 0 },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uMistColor: { value: new THREE.Color('#d2e4f0') },
      uFogDensity: { value: 0.00044 },
      uMaxElev: { value: maxElev },
      uVScale: { value: 1.0 },
      uStyleMode: { value: 0 },
      uTime: { value: 0.0 },
      uFocusPos: { value: new THREE.Vector3(0, 0, 0) },
      // World XZ of the height / mask / climate textures' centre (the seam atlas follows the player)
      uTileOrigin: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: /* glsl */ `
      attribute vec4 aRingData; // (stepSize, snapPeriod, edgeSnapX, edgeSnapZ)

      uniform sampler2D uHeightTex;
      uniform float uWorldSpan;
      uniform float uGridSize;
      uniform float uCellReg;
      uniform float uVScale;
      uniform vec3 uFocusPos;
      uniform vec2 uTileOrigin;

      varying vec2 vUv;
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vElevRaw;

      vec2 atlasSpan() {
        return vec2(uWorldSpan);
      }
      vec2 atlasGrid() {
        return vec2(uGridSize);
      }

      // xz is LOCAL to the tile/atlas (relative to uTileOrigin)
      vec2 localToHeightUv(vec2 localXZ) {
        vec2 span = atlasSpan();
        vec2 grid = atlasGrid();
        vec2 uv = clamp(localXZ / span + 0.5, 0.0, 1.0);
        uv.y = 1.0 - uv.y;
        return mix((uv * (grid - 1.0) + 0.5) / grid, uv, uCellReg);
      }

      float sampleRawH(vec2 localXZ) {
        return texture2D(uHeightTex, localToHeightUv(localXZ)).r;
      }

      void main() {
        float stepSize = aRingData.x;
        float snapPeriod = max(aRingData.y, 1.0);
        vec2 edgeSnap = aRingData.zw;
        // Clipmap covers a square of uWorldSpan; atlas may be rectangular inside it
        float halfSpan = uWorldSpan * 0.5;
        vec2 span = atlasSpan();
        vec2 halfAtlas = span * 0.5;

        // Uniform-snap clipmap center in LOCAL tile space
        vec2 focusLocal = uFocusPos.xz - uTileOrigin;
        vec2 ringCenter = floor(focusLocal / snapPeriod + 0.5) * snapPeriod;
        // A cell-registered raster has its samples at cell centres, half a cell from where the
        // grid would put vertices. Put the vertices ON the samples: otherwise each vertex shows
        // the average of four samples, which shaves metres off sharp ridges and cliff tops and
        // leaves everything standing on the true height (the player, grass) floating above.
        ringCenter += uCellReg * 0.5 * span / atlasGrid();
        vec2 rawLocal = ringCenter + position.xz;
        vec2 localXZ = clamp(rawLocal, -halfSpan, halfSpan);
        // Height samples clamp to atlas bounds
        vec2 sampleXZ = clamp(localXZ, -halfAtlas, halfAtlas);

        float hRaw;
        if (edgeSnap.x > 0.0 || edgeSnap.y > 0.0) {
          vec2 xzA = clamp(rawLocal - edgeSnap, -halfAtlas, halfAtlas);
          vec2 xzB = clamp(rawLocal + edgeSnap, -halfAtlas, halfAtlas);
          hRaw = 0.5 * (sampleRawH(xzA) + sampleRawH(xzB));
        } else {
          hRaw = sampleRawH(sampleXZ);
        }

        vec2 planeUv = clamp(sampleXZ / span + 0.5, 0.0, 1.0);
        planeUv.y = 1.0 - planeUv.y;
        vUv = planeUv;
        vElevRaw = hRaw;

        vec3 worldP = vec3(localXZ.x + uTileOrigin.x, hRaw * uVScale, localXZ.y + uTileOrigin.y);
        vWorldPos = worldP;

        float eps = max(stepSize, span.x / max(atlasGrid().x - 1.0, 1.0));
        float hL = sampleRawH(clamp(sampleXZ - vec2(eps, 0.0), -halfAtlas, halfAtlas)) * uVScale;
        float hR = sampleRawH(clamp(sampleXZ + vec2(eps, 0.0), -halfAtlas, halfAtlas)) * uVScale;
        float hD = sampleRawH(clamp(sampleXZ - vec2(0.0, eps), -halfAtlas, halfAtlas)) * uVScale;
        float hU = sampleRawH(clamp(sampleXZ + vec2(0.0, eps), -halfAtlas, halfAtlas)) * uVScale;
        vNormalW = normalize(vec3(hL - hR, 2.0 * eps, hD - hU));

        gl_Position = projectionMatrix * viewMatrix * vec4(worldP, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uMasks;
      uniform sampler2D uHeightTex;
      uniform sampler2D uClimate;
      uniform int uHasClimate;
      uniform sampler2D uRock;
      uniform int uHasRock;
      uniform vec2 uTempRange;
      uniform vec2 uPrecipRange;
      uniform float uWorldSpan;
      uniform float uGridSize;
      uniform float uCellReg;
      uniform vec2 uTileOrigin;
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform vec3 uMistColor;
      uniform float uFogDensity;
      uniform float uMaxElev;
      uniform float uVScale;
      uniform int uStyleMode;
      uniform float uTime;

      varying vec2 vUv;
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vElevRaw;

      vec2 fragAtlasSpan() {
        return vec2(uWorldSpan);
      }
      vec2 fragAtlasGrid() {
        return vec2(uGridSize);
      }

      // ---------- Fast Procedural Noise ----------
      float hash21(vec2 p) {
        p = fract(p * vec2(127.1, 311.7));
        p += dot(p, p + 34.53);
        return fract(p.x * p.y);
      }

      float vnoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        float a = hash21(i);
        float b = hash21(i + vec2(1.0, 0.0));
        float c = hash21(i + vec2(0.0, 1.0));
        float d = hash21(i + vec2(1.0, 1.0));
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
      }

      float fbm3(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
        for (int i = 0; i < 3; i++) {
          v += a * vnoise(p);
          p = rot * p * 2.04;
          a *= 0.5;
        }
        return v;
      }

      // Ridged craggy rock height for triplanar cliff detail
      float rockPattern(vec2 p) {
        float strata = sin(p.y * 2.6 + fbm3(p * 0.6) * 3.8) * 0.5 + 0.5;
        float n1 = 1.0 - abs(vnoise(p * 1.4) * 2.0 - 1.0);
        float n2 = 1.0 - abs(vnoise(p * 3.7 + vec2(4.2, 1.7)) * 2.0 - 1.0);
        float cracks = pow(n1, 1.8) * 0.55 + pow(n2, 2.2) * 0.30;
        return mix(cracks, strata, 0.38);
      }

      // Triplanar scalar rock height + finite-difference normal perturbation
      vec3 perturbTriplanarNormal(vec3 worldPos, vec3 baseN, float rockAmt, float groundAmt) {
        vec3 w = pow(abs(baseN), vec3(4.0));
        w /= max(w.x + w.y + w.z, 0.0001);

        float scale = 0.14;
        float eps = 0.35;

        // Sample triplanar rock height around point
        float h0 = rockPattern(worldPos.zy * scale) * w.x
                 + rockPattern(worldPos.xz * scale) * w.y
                 + rockPattern(worldPos.xy * scale) * w.z;

        float hx = rockPattern((worldPos.zy + vec2(0.0, 0.0)) * scale) * w.x
                 + rockPattern((worldPos.xz + vec2(eps, 0.0)) * scale) * w.y
                 + rockPattern((worldPos.xy + vec2(eps, 0.0)) * scale) * w.z;

        float hy = rockPattern((worldPos.zy + vec2(0.0, eps)) * scale) * w.x
                 + rockPattern((worldPos.xz) * scale) * w.y
                 + rockPattern((worldPos.xy + vec2(0.0, eps)) * scale) * w.z;

        float hz = rockPattern((worldPos.zy + vec2(eps, 0.0)) * scale) * w.x
                 + rockPattern((worldPos.xz + vec2(0.0, eps)) * scale) * w.y
                 + rockPattern((worldPos.xy) * scale) * w.z;

        vec3 rockGrad = vec3(hx - h0, hy - h0, hz - h0) / eps;

        // Subtle high-frequency soil/grass micro-bump on flat terrain
        float gScale = 0.42;
        float g0 = fbm3(worldPos.xz * gScale);
        float gx = fbm3((worldPos.xz + vec2(0.4, 0.0)) * gScale);
        float gz = fbm3((worldPos.xz + vec2(0.0, 0.4)) * gScale);
        vec3 groundGrad = vec3(gx - g0, 0.0, gz - g0) * 1.1;

        vec3 grad = rockGrad * (1.85 * rockAmt) + groundGrad * (0.65 * groundAmt);
        // Project gradient tangent to base normal
        grad -= baseN * dot(grad, baseN);
        return normalize(baseN - grad);
      }

      // Sample exact terrain height (meters * vScale); xz is WORLD, converted via uTileOrigin
      float sampleHeightAtWorld(vec2 xz) {
        vec2 span = fragAtlasSpan();
        vec2 grid = fragAtlasGrid();
        vec2 localXZ = xz - uTileOrigin;
        vec2 uv = localXZ / span + 0.5;
        uv.y = 1.0 - uv.y;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return -50.0;
        vec2 texUv = mix((uv * (grid - 1.0) + 0.5) / grid, uv, uCellReg);
        return texture2D(uHeightTex, texUv).r * uVScale;
      }

      // Full-resolution analytical heightfield normal (local to this tile/atlas)
      vec3 computeHeightTexNormal(vec2 xz) {
        vec2 span = fragAtlasSpan();
        vec2 grid = fragAtlasGrid();
        float eps = span.x / max(grid.x - 1.0, 1.0);
        vec2 halfAtlas = span * 0.5;
        vec2 localXZ = xz - uTileOrigin;
        vec2 cL = clamp(localXZ - vec2(eps, 0.0), -halfAtlas, halfAtlas) + uTileOrigin;
        vec2 cR = clamp(localXZ + vec2(eps, 0.0), -halfAtlas, halfAtlas) + uTileOrigin;
        vec2 cD = clamp(localXZ - vec2(0.0, eps), -halfAtlas, halfAtlas) + uTileOrigin;
        vec2 cU = clamp(localXZ + vec2(0.0, eps), -halfAtlas, halfAtlas) + uTileOrigin;
        float hL = sampleHeightAtWorld(cL);
        float hR = sampleHeightAtWorld(cR);
        float hD = sampleHeightAtWorld(cD);
        float hU = sampleHeightAtWorld(cU);
        return normalize(vec3(hL - hR, 2.0 * eps, hD - hU));
      }

      // Real mountain self-shadowing via heightfield raymarch toward sun
      float computeMountainShadow(vec3 wp, vec3 sunDir) {
        if (sunDir.y <= 0.05) return 0.25;
        float shadow = 1.0;
        float t = 10.0;
        for (int i = 0; i < 14; i++) {
          vec3 samplePos = wp + sunDir * t;
          float h = sampleHeightAtWorld(samplePos.xz);
          float diff = h - samplePos.y;
          if (diff > 0.0) {
            // Soft penumbra falloff with distance
            float penumbra = clamp(1.0 - (diff / (t * 0.065 + 2.5)), 0.12, 1.0);
            shadow = min(shadow, penumbra);
            if (shadow < 0.14) break;
          }
          t *= 1.38;
          if (t > 900.0) break;
        }
        return shadow;
      }

      void main() {
        vec4 m = texture2D(uMasks, vUv);
        float riverWet = m.r;
        float lake = m.g;
        float curvAO = m.b;     // 0 = deep gully, 0.5 = flat, 1 = sharp ridge
        float moisture = m.a;

        vec3 nTex = computeHeightTexNormal(vWorldPos.xz);
        vec3 nSmooth = normalize(mix(normalize(vNormalW), nTex, 0.88));
        vec3 nFlat = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        vec3 baseN = (uStyleMode == 1) ? nFlat : nSmooth;

        float baseSlope = 1.0 - clamp(baseN.y, 0.0, 1.0);
        float elev = vElevRaw;
        float hNorm = clamp(elev / max(uMaxElev, 1.0), 0.0, 1.0);

        // Triplanar rock & soil detail normal in High-Detail mode
        float rockMask = smoothstep(0.08, 0.22, baseSlope) * smoothstep(1.2, 6.5, elev);
        float groundMask = (1.0 - rockMask) * smoothstep(2.0, 7.0, elev);
        vec3 N = (uStyleMode == 0)
          ? perturbTriplanarNormal(vWorldPos, baseN, rockMask, groundMask)
          : baseN;

        // Multi-scale painterly/natural ground variation
        float macroNoise = fbm3(vWorldPos.xz * 0.008);
        float midNoise = fbm3(vWorldPos.xz * 0.045);
        float fineNoise = vnoise(vWorldPos.xz * 0.35);

        // --- Rich Natural Albedo Palette (Climate-Driven when WorldClim data is present!) ---
        vec3 seabed     = vec3(0.08, 0.19, 0.28);
        vec3 wetSand    = vec3(0.55, 0.49, 0.36);
        vec3 drySand    = vec3(0.82, 0.75, 0.56);
        vec3 aridSteppe = vec3(0.71, 0.62, 0.38);
        vec3 warmScrub  = vec3(0.56, 0.55, 0.30);
        vec3 meadow     = vec3(0.30, 0.51, 0.22);
        vec3 lushMoss   = vec3(0.16, 0.37, 0.18);
        vec3 borealPine = vec3(0.19, 0.31, 0.19);
        vec3 alpineTund = vec3(0.42, 0.44, 0.32);

        // Layered sedimentary & granite rock colors
        vec3 rockDark   = vec3(0.24, 0.24, 0.27);
        vec3 rockWarm   = vec3(0.52, 0.43, 0.33);
        vec3 rockSlate  = vec3(0.37, 0.41, 0.46);
        vec3 snowCol    = vec3(0.95, 0.97, 1.00);

        float tempC = 14.0 - hNorm * 18.0;
        float precipNorm = moisture;
        if (uHasClimate == 1) {
          vec4 cl = texture2D(uClimate, vUv);
          tempC = mix(uTempRange.x, uTempRange.y, cl.r);
          float precipMm = mix(uPrecipRange.x, uPrecipRange.y, cl.g);
          precipNorm = clamp(precipMm / 650.0, 0.0, 1.0);
        }

        // Bedrock: how much of each rock type lies under this spot. The lookup is nudged by
        // noise so the coarse map's boundaries come out ragged, and the weights are sharpened
        // so a hillside reads as one rock, not a blend.
        float wGranite = 0.0;
        float wVolcanic = 0.0;
        float wLime = 0.0;
        float wMeta = 0.0;
        float youngLava = 0.0;
        float saltPan = 0.0;
        float sinter = 0.0;
        if (uHasRock == 1) {
          vec2 nudge = (vec2(fbm3(vWorldPos.xz * 0.011), fbm3(vWorldPos.zx * 0.011 + 31.7)) - 0.5) * (90.0 / uWorldSpan);
          vec4 rk = texture2D(uRock, clamp(vUv + nudge, 0.0, 1.0));
          // Cones are small and exact: read them without the nudge
          vec4 rkExact = texture2D(uRock, vUv);
          youngLava = smoothstep(0.84, 0.98, rkExact.g);
          // Likewise a dry basin floor (limestone channel above the regional cap, see basins.py)
          saltPan = smoothstep(0.84, 0.98, rkExact.b);
          // ...and the mineral crust around a hot spring (granite channel, see volcano.py)
          sinter = smoothstep(0.84, 0.98, rkExact.r);
          wGranite = smoothstep(0.18, 0.5, rk.r);
          wVolcanic = max(smoothstep(0.18, 0.5, rk.g), youngLava);
          wLime = smoothstep(0.18, 0.5, rk.b);
          wMeta = smoothstep(0.18, 0.5, rk.a);
          float wSum = max(1.0, wGranite + wVolcanic + wLime + wMeta);
          wGranite /= wSum; wVolcanic /= wSum; wLime /= wSum; wMeta /= wSum;
        }
        float wRock = wGranite + wVolcanic + wLime + wMeta;

        // Biome ground blend driven by temperature + precipitation + riparian wetness
        float effMoist = clamp(mix(moisture, precipNorm, uHasClimate == 1 ? 0.55 : 0.0) + riverWet * 0.45, 0.0, 1.0);
        vec3 warmGround = mix(aridSteppe, warmScrub, smoothstep(0.18, 0.48, effMoist));
        vec3 lushGround = mix(meadow, lushMoss, smoothstep(0.52, 0.88, effMoist));
        vec3 desertSand  = vec3(0.87, 0.72, 0.46);
        vec3 desertDune  = vec3(0.79, 0.59, 0.34);
        vec3 redMesa     = vec3(0.64, 0.33, 0.21);
        vec3 tundraMoss  = vec3(0.46, 0.43, 0.29);
        vec3 glacierIce  = vec3(0.73, 0.88, 0.96);

        float desertMask = 0.0;
        float glacierMask = 0.0;
        vec3 grassCol = mix(warmGround, lushGround, smoothstep(0.30, 0.65, effMoist));
        if (uHasClimate == 1) {
          // True Hot Sand Desert (Sahara / Namib / Arabian ergs with subtle wind ripples!)
          desertMask = smoothstep(12.0, 20.0, tempC) * (1.0 - smoothstep(0.16, 0.46, precipNorm));
          float duneRipple = 0.5 + 0.5 * sin(vWorldPos.x * 0.16 + vWorldPos.z * 0.09 + macroNoise * 4.5);
          vec3 duneCol = mix(desertDune, desertSand, duneRipple * 0.55 + midNoise * 0.45);
          // Warm dry steppe -> true golden desert sand
          float warmDry = smoothstep(9.0, 17.0, tempC) * (1.0 - smoothstep(0.22, 0.54, precipNorm));
          grassCol = mix(grassCol, aridSteppe * 1.04, warmDry * 0.78);
          grassCol = mix(grassCol, duneCol, desertMask * 0.95);
          // Cool boreal pine -> mossy lichen tundra
          float coolBoreal = (1.0 - smoothstep(2.0, 9.5, tempC)) * smoothstep(0.25, 0.65, effMoist);
          grassCol = mix(grassCol, borealPine, coolBoreal * 0.62);
          float coldTundra = 1.0 - smoothstep(-1.5, 4.2, tempC);
          grassCol = mix(grassCol, mix(tundraMoss, alpineTund, midNoise), coldTundra * 0.85);
          // Polar frozen ice sheet / glacier (right down to sea level!)
          glacierMask = 1.0 - smoothstep(-5.5, -1.8, tempC);
          grassCol = mix(grassCol, mix(snowCol * 0.94, glacierIce, midNoise * 0.35), glacierMask * 0.96);
        } else {
          grassCol = mix(grassCol, alpineTund, smoothstep(0.42, 0.72, hNorm));
        }
        // Natural soil & clover mottling at 3 scales (breaks up flat green)
        grassCol = mix(grassCol * 0.84, grassCol * 1.13, macroNoise * 0.55 + midNoise * 0.30 + fineNoise * 0.15);

        // Beach transition (wet shore -> dry sand -> grass/desert/glacier)
        vec3 sandCol = mix(wetSand, drySand, smoothstep(0.3, 2.8, elev));
        // Beaches are ground-up bedrock: black on volcanic coasts, near-white on limestone
        sandCol = mix(sandCol, vec3(0.15, 0.14, 0.15), wVolcanic * 0.85);
        sandCol = mix(sandCol, vec3(0.93, 0.90, 0.80), wLime * 0.6);
        sandCol = mix(sandCol, glacierIce * 0.95, glacierMask * 0.85);
        vec3 albedo = mix(seabed, sandCol, smoothstep(-5.0, 0.8, elev));
        albedo = mix(albedo, grassCol, smoothstep(2.0, 7.5, elev));

        // Detailed stratified rock on slopes & high peaks (red sandstone mesas in deserts, glacial blue ice in polar!)
        float strataBand = rockPattern(vec2(vWorldPos.x * 0.08 + vWorldPos.z * 0.08, vWorldPos.y * 0.14));
        float coldRockT = (uHasClimate == 1) ? (1.0 - smoothstep(0.0, 14.0, tempC)) : smoothstep(0.30, 0.78, hNorm);
        vec3 rockCol = mix(rockWarm, rockSlate, coldRockT);
        rockCol = mix(rockCol, redMesa, desertMask * 0.78);
        rockCol = mix(rockDark, rockCol * 1.16, strataBand);
        if (wRock > 0.01) {
          vec3 wp = vWorldPos;
          // Stripes finer than a pixel shimmer into moiré from afar: fade each pattern to its
          // average as its wavelength nears the size of a pixel on the ground
          vec3 fw = fwidth(wp);
          float pxSize = max(fw.x, max(fw.y, fw.z));
          float fineFade = 1.0 - smoothstep(0.25, 1.1, pxSize * 2.0);
          float bandFade = 1.0 - smoothstep(0.25, 1.1, pxSize * 1.2);
          float speck = mix(0.5, vnoise(wp.xz * 2.3 + wp.y * 1.7), fineFade);
          float joints = rockPattern(vec2(wp.x * 0.05 - wp.z * 0.03, wp.y * 0.05 + wp.z * 0.04));
          // Granite: pale, speckled, broken by widely spaced joints into rounded blocks
          vec3 granite = mix(vec3(0.40, 0.37, 0.36), vec3(0.74, 0.69, 0.65), joints * 0.7 + speck * 0.3);
          granite = mix(granite, vec3(0.80, 0.66, 0.60), smoothstep(0.55, 0.9, midNoise) * 0.35);
          // Basalt: near black, in vertical columns on cliffs and rubbly on gentle ground
          float columns = 0.5 + 0.5 * fineFade * sin((wp.x * 0.83 + wp.z * 0.71) * 1.9 + fbm3(wp.xz * 0.05) * 5.0);
          vec3 basalt = mix(vec3(0.09, 0.09, 0.10), vec3(0.25, 0.23, 0.22), mix(speck, columns, smoothstep(0.25, 0.5, baseSlope) * 0.75));
          // A young cone is loose cinder: rust red where it oxidised, black ash elsewhere
          vec3 cinder = mix(vec3(0.11, 0.10, 0.10), vec3(0.36, 0.18, 0.13), smoothstep(0.42, 0.78, fbm3(wp.xz * 0.02 + 11.0)));
          basalt = mix(basalt, cinder * (0.8 + 0.4 * speck), youngLava);
          // Limestone: cream to pale grey in thin, even, level beds, streaked dark by rain
          float beds = 0.5 + 0.5 * bandFade * sin(wp.y * 1.15 + fbm3(wp.xz * 0.02) * 2.4);
          float streaks = fbm3(vec2((wp.x + wp.z) * 0.4, wp.y * 0.05));
          vec3 lime = mix(vec3(0.62, 0.61, 0.57), vec3(0.90, 0.87, 0.78), beds * 0.6 + 0.4) * mix(0.78, 1.0, streaks);
          // Slate and schist: blue-grey sheets tilted steeply, with a faint sheen
          float folia = 0.5 + 0.5 * bandFade * sin(dot(wp, vec3(0.55, 0.80, 0.35)) * 1.5 + fbm3(wp.xz * 0.03) * 3.0);
          vec3 slate = mix(vec3(0.17, 0.20, 0.25), vec3(0.42, 0.47, 0.54), folia);
          vec3 typed = (granite * wGranite + basalt * wVolcanic + lime * wLime + slate * wMeta) / wRock;
          rockCol = mix(rockCol, typed, wRock);
        }
        // Darken rock crevices using fine rock noise + cavity AO
        rockCol *= (0.72 + 0.42 * curvAO);
        // Steep slopes on polar ice sheets expose gleaming pale-blue glacier ice + dark nunatak rock
        vec3 glacialSlope = mix(glacierIce, rockCol, smoothstep(0.28, 0.52, baseSlope));
        rockCol = mix(rockCol, glacialSlope, glacierMask * 0.82);
        // High-altitude rocky scree even on moderate slopes (suppressed on flat sandy deserts & glaciers)
        float screeMask = clamp(rockMask * (1.0 - desertMask * 0.45) + smoothstep(0.58, 0.82, hNorm) * 0.65 * (1.0 - desertMask * 0.6), 0.0, 1.0);
        // Granite and limestone crop out through thin soil; a young cone is bare all over
        screeMask = clamp(screeMask + (wGranite + wLime) * 0.22 * smoothstep(0.03, 0.14, baseSlope), 0.0, 1.0);
        screeMask = max(screeMask, youngLava * smoothstep(1.0, 4.0, elev) * 0.94);
        // Thin or absent plant cover lets the bedrock colour the ground itself
        float bareGround = clamp(desertMask * 0.55 + (1.0 - smoothstep(0.15, 0.5, effMoist)) * 0.25, 0.0, 1.0) * wRock;
        albedo = mix(albedo, mix(albedo, rockCol, 0.6), bareGround * smoothstep(2.0, 7.5, elev));
        albedo = mix(albedo, rockCol, screeMask);
        // Salt flat: blinding white crust broken into polygons by darker, damp cracks
        if (saltPan > 0.01) {
          float cell = vnoise(vWorldPos.xz * 0.22) * 0.6 + vnoise(vWorldPos.xz * 0.51 + 9.3) * 0.4;
          float crackLine = 1.0 - smoothstep(0.0, 0.05, abs(cell - 0.5));
          vec3 salt = mix(vec3(0.93, 0.92, 0.88), vec3(0.80, 0.76, 0.68), macroNoise * 0.6);
          salt = mix(salt, vec3(0.55, 0.50, 0.44), crackLine * 0.55);
          albedo = mix(albedo, salt, saltPan * smoothstep(0.2, 0.6, elev));
        }
        // Hot spring: pale silica crust, stained orange and rust where warm water runs over it
        if (sinter > 0.01) {
          vec3 crust = mix(vec3(0.90, 0.88, 0.82), vec3(0.74, 0.70, 0.62), macroNoise * 0.6);
          float mats = smoothstep(0.45, 0.68, vnoise(vWorldPos.xz * 0.07) * 0.6 + vnoise(vWorldPos.xz * 0.27 + 4.0) * 0.4);
          crust = mix(crust, mix(vec3(0.86, 0.52, 0.12), vec3(0.56, 0.25, 0.08), fineNoise), mats * 0.75);
          albedo = mix(albedo, crust, sinter);
        }

        // Snow & Polar Ice accumulation (polar ice sheets reach sea level; temperate snow follows freezing isotherm)
        float snowMask = 0.0;
        if (uHasClimate == 1) {
          float freezeT = 1.6 + (macroNoise - 0.5) * 2.2 + precipNorm * 1.4;
          float alpineSnow = (1.0 - smoothstep(freezeT - 2.6, freezeT, tempC)) * smoothstep(14.0, 55.0, elev);
          float polarSnow = glacierMask * smoothstep(0.4, 3.5, elev);
          snowMask = max(alpineSnow, polarSnow);
          snowMask *= (1.0 - smoothstep(0.22, 0.44, baseSlope));
        } else {
          float snowLine = 0.62 + (macroNoise - 0.5) * 0.10 + (1.0 - moisture) * 0.06;
          snowMask = smoothstep(snowLine, snowLine + 0.13, hNorm) * (1.0 - smoothstep(0.18, 0.36, baseSlope));
        }
        snowMask *= 1.0 - sinter; // warm ground around a hot spring holds no snow
        albedo = mix(albedo, snowCol, snowMask);

        // Wet riverbanks & gravel bed under the 3D flowing river ribbons
        float wetSpec = 0.0;
        if (riverWet > 0.05 && elev > -1.0) {
          vec3 wetPebbles = mix(vec3(0.24, 0.23, 0.20), vec3(0.38, 0.35, 0.30), fineNoise);
          float bankMask = smoothstep(0.06, 0.45, riverWet);
          albedo = mix(albedo, wetPebbles, bankMask * 0.82);
          wetSpec = bankMask * 0.45;
        }
        // Underwater lakebeds get dark rocky silt beneath the real 3D flat lake mesh
        if (lake > 0.35 && elev > 0.0) {
          vec3 lakeBed = mix(vec3(0.22, 0.24, 0.21), vec3(0.12, 0.16, 0.18), fineNoise);
          lakeBed = mix(lakeBed, vec3(0.84, 0.56, 0.14), sinter); // a hot pool's bed is bright with mats
          albedo = mix(albedo, lakeBed, smoothstep(0.35, 0.7, lake) * 0.85);
          wetSpec = max(wetSpec, 0.35);
        }

        // Geological cavity AO (darkens gullies & river valleys, brightens exposed ridges)
        float ao = mix(0.66, 1.12, curvAO);

        // --- Lighting & Real Cast Shadows ---
        vec3 sunDir = normalize(uSunDir);
        float NdotL = max(dot(N, sunDir), 0.0);
        vec3 sunColor = mix(vec3(1.0, 0.72, 0.46), vec3(1.0, 0.96, 0.89), smoothstep(0.10, 0.52, sunDir.y));
        vec3 skyAmbient = mix(vec3(0.28, 0.36, 0.52), vec3(0.42, 0.58, 0.80), smoothstep(0.10, 0.52, sunDir.y));
        vec3 bounceLight = vec3(0.26, 0.24, 0.18) * (0.5 - 0.5 * N.y);

        vec3 color;
        if (uStyleMode == 0) {
          // Real ray-marched mountain cast shadows + gentle drifting cloud shadows
          float mtnShadow = computeMountainShadow(vWorldPos, sunDir);
          float cloudVal = fbm3(vWorldPos.xz * 0.0018 + vec2(uTime * 0.008, uTime * 0.004));
          float cloudShadow = mix(0.74, 1.0, smoothstep(0.38, 0.62, cloudVal));

          float directShadow = mtnShadow * cloudShadow;
          vec3 direct = sunColor * NdotL * directShadow * 1.38;
          vec3 ambient = (skyAmbient * (0.42 + 0.58 * max(N.y, 0.0)) + bounceLight) * ao * 0.68;

          // Subtle Blinn-Phong specular on wet riverbanks, snow & sunlit rock
          vec3 V = normalize(cameraPosition - vWorldPos);
          vec3 H = normalize(sunDir + V);
          float specPow = mix(24.0, 64.0, wetSpec);
          float specAmt = (wetSpec * 0.65 + snowMask * 0.22 + rockMask * (0.08 + wMeta * 0.16));
          vec3 spec = sunColor * pow(max(dot(N, H), 0.0), specPow) * specAmt * directShadow;

          color = albedo * (ambient + direct) + spec;
        } else if (uStyleMode == 1) {
          // Crisp Faceted Diorama mode
          float mtnShadow = computeMountainShadow(vWorldPos, sunDir);
          float diff = NdotL * mix(0.45, 1.0, mtnShadow);
          float stepped = floor(diff * 4.0 + 0.5) / 4.0;
          float lit = mix(diff, stepped, 0.40);
          vec3 light = mix(vec3(0.44, 0.54, 0.72), sunColor * 1.15, lit);
          float lum = dot(albedo, vec3(0.299, 0.587, 0.114));
          vec3 satAlbedo = mix(vec3(lum), albedo, 1.16);
          color = satAlbedo * light * mix(0.85, 1.08, curvAO);
        } else {
          // Topographic / Geomorphology Atlas mode
          float diff = 0.42 + 0.58 * NdotL;
          vec3 topoBase = mix(albedo, vec3(0.86, 0.88, 0.84), 0.35) * diff;
          float cliffWarn = smoothstep(0.14, 0.24, baseSlope) * smoothstep(2.0, 6.0, elev);
          topoBase = mix(topoBase, vec3(0.92, 0.34, 0.24), cliffWarn * 0.58);
          float elevScaled = vWorldPos.y;
          if (elev > 0.5) {
            float minor = abs(fract(elevScaled / 10.0 - 0.5) - 0.5) / max(fwidth(elevScaled / 10.0), 0.0001);
            float major = abs(fract(elevScaled / 50.0 - 0.5) - 0.5) / max(fwidth(elevScaled / 50.0), 0.0001);
            topoBase = mix(topoBase, vec3(0.14, 0.18, 0.22), (1.0 - smoothstep(0.6, 1.5, minor)) * 0.30);
            topoBase = mix(topoBase, vec3(0.06, 0.08, 0.12), (1.0 - smoothstep(0.8, 1.8, major)) * 0.70);
          }
          topoBase = mix(topoBase, vec3(0.06, 0.62, 0.98), smoothstep(0.18, 0.45, max(riverWet, lake)) * 0.88);
          color = topoBase;
        }

        // --- Height-Based Valley Mist + Aerial Perspective ---
        float dist = length(cameraPosition - vWorldPos);
        if (uStyleMode == 0) {
          // Valley mist settles gently in lowlands & fades with altitude so peaks pop
          float heightFactor = exp(-max(vWorldPos.y, 0.0) * 0.014);
          float valleyMist = (1.0 - exp(-dist * 0.00065)) * heightFactor * 0.48;
          color = mix(color, uMistColor, clamp(valleyMist, 0.0, 0.55));

          // Distant aerial perspective (cool blue shift with distance)
          float aerial = 1.0 - exp(-pow(dist * uFogDensity, 1.5));
          color = mix(color, uFogColor, clamp(aerial, 0.0, 0.90));
        } else {
          float density = (uStyleMode == 1) ? uFogDensity * 0.34 : uFogDensity * 0.22;
          float fogFactor = 1.0 - exp(-pow(dist * density, 1.5));
          color = mix(color, uFogColor, clamp(fogFactor, 0.0, 0.88));
        }

        // Not-yet-synthesized "ghost" tiles (climate alpha 0) read as hazy, washed-out terrain
        float ghostAmt = 1.0 - texture2D(uClimate, vUv).a;
        if (ghostAmt > 0.01) {
          vec3 washed = mix(vec3(dot(color, vec3(0.30, 0.59, 0.11))), uMistColor, 0.42);
          color = mix(color, washed, ghostAmt * 0.72);
        }
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    extensions: {
      derivatives: true,
    },
  });
}

/**
 * Animated Ocean Water Plane with exact Float32 depth-based turquoise shallows,
 * moving wave ripples, Fresnel sky reflection, and shoreline foam.
 */
export function createWaterMaterial(heightFloatTex, maskTex, worldSpan, gridSize = 1024) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uHeightTex: { value: heightFloatTex },
      uMasks: { value: maskTex },
      // Span, texel count and world-XZ centre of the height / mask textures
      uWorldSpan: { value: worldSpan },
      uGridSize: { value: gridSize },
      uCellReg: { value: 0 },
      uTexOrigin: { value: new THREE.Vector2(0, 0) },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uFogDensity: { value: 0.00044 },
      uTime: { value: 0.0 },
      uStyleMode: { value: 0 },
      uVScale: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      uniform float uTime;
      varying vec3 vWorldPos;

      void main() {
        vec3 pos = position;
        // A slow swell (9 to 12 s per wave): faster reads as the whole sea sloshing in and out
        pos.y += sin(pos.x * 0.045 + uTime * 0.68) * 0.24 + cos(pos.z * 0.055 - uTime * 0.54) * 0.20;
        vec4 wp = modelMatrix * vec4(pos, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uHeightTex;
      uniform sampler2D uMasks;
      uniform float uWorldSpan;
      uniform float uGridSize;
      uniform float uCellReg;
      uniform vec2 uTexOrigin;
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform float uTime;
      uniform int uStyleMode;
      uniform float uVScale;

      varying vec3 vWorldPos;

      float hash21(vec2 p) {
        p = fract(p * vec2(127.1, 311.7));
        p += dot(p, p + 34.53);
        return fract(p.x * p.y);
      }

      float vnoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
                   mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
      }

      void main() {
        vec2 uv = (vWorldPos.xz - uTexOrigin) / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y;
        bool inMap = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
        vec2 texUv = mix((uv * (uGridSize - 1.0) + 0.5) / uGridSize, uv, uCellReg);
        // Never draw the y=0 ocean sheet underneath an inland crater lake whose floor dips below sea level!
        if (inMap && texture2D(uMasks, texUv).g > 0.05) discard;
        float terrainH = inMap
          ? texture2D(uHeightTex, texUv).r * uVScale
          : -30.0;

        float depthM = max(-terrainH, 0.0);

        // Dual scrolling wave normals
        vec2 p1 = vWorldPos.xz * 0.14 + vec2(uTime * 0.45, uTime * 0.28);
        vec2 p2 = vWorldPos.xz * 0.27 - vec2(uTime * 0.32, -uTime * 0.41);
        float w1 = vnoise(p1);
        float w2 = vnoise(p2);
        vec3 N = normalize(vec3((w1 - 0.5) * 0.24 + (w2 - 0.5) * 0.14, 1.0, (w2 - 0.5) * 0.24 - (w1 - 0.5) * 0.14));

        vec3 shallowCol = vec3(0.14, 0.62, 0.66);
        vec3 deepCol    = vec3(0.05, 0.20, 0.38);
        vec3 baseCol = mix(shallowCol, deepCol, smoothstep(0.0, 16.0, depthM));

        // Animated shoreline foam bands
        float shoreMask = 1.0 - smoothstep(0.0, 3.8, depthM);
        float foamRipple = sin(depthM * 3.8 - uTime * 1.1 + w1 * 3.0) * 0.5 + 0.5;
        float foam = smoothstep(0.42, 0.72, shoreMask + foamRipple * shoreMask * 0.5);
        baseCol = mix(baseCol, vec3(0.92, 0.97, 1.0), foam * 0.85);

        // Fresnel sky reflection + crisp sun sparkle
        vec3 V = normalize(cameraPosition - vWorldPos);
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.4);
        baseCol = mix(baseCol, uFogColor * 1.05, fresnel * 0.58);

        vec3 H = normalize(normalize(uSunDir) + V);
        float spec = pow(max(dot(N, H), 0.0), 96.0);
        baseCol += vec3(1.0, 0.96, 0.86) * spec * 0.95;

        float dist = length(cameraPosition - vWorldPos);
        float density = (uStyleMode == 0) ? uFogDensity : uFogDensity * 0.35;
        float fogFactor = 1.0 - exp(-pow(dist * density, 1.5));
        baseCol = mix(baseCol, uFogColor, clamp(fogFactor, 0.0, 0.90));

        float alpha = mix(0.65, 0.95, smoothstep(0.0, 4.0, depthM));
        gl_FragColor = vec4(baseCol, alpha);
      }
    `,
  });
}

/**
 * Sky dome with atmospheric horizon glow & sun disc.
 */
export function createSkyDome() {
  const geo = new THREE.SphereGeometry(5200, 32, 20);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uZenith: { value: new THREE.Color('#276bc4') },
      uHorizon: { value: new THREE.Color('#b6dcf5') },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform vec3 uZenith;
      uniform vec3 uHorizon;
      varying vec3 vDir;

      void main() {
        vec3 d = normalize(vDir);
        float h = clamp(d.y, 0.0, 1.0);
        vec3 col = mix(uHorizon, uZenith, pow(h, 0.52));

        float sunDot = max(dot(d, normalize(uSunDir)), 0.0);
        col += vec3(1.0, 0.86, 0.62) * pow(sunDot, 10.0) * 0.38;
        col += vec3(1.0, 0.98, 0.90) * smoothstep(0.9985, 0.9994, sunDot) * 1.6;

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  return new THREE.Mesh(geo, mat);
}
