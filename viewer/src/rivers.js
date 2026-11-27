import * as THREE from 'three';

/**
 * Builds real 3D flowing river ribbon meshes along physically solved
 * lake-to-sea and alpine-spring-to-sea drainage splines.
 *
 * Physical guarantees:
 *  - Every cross-section is strictly HORIZONTALLY LEVEL (water never tilts or climbs hillsides!)
 *  - Left and right half-widths are independently bank-confined against the actual rendered
 *    terrain mesh so rivers hug steep canyon bluffs on one side while widening across flat
 *    valley floodplains on the other.
 *  - Centerline elevation receives a small capped clearance lift (<= +0.60 m) above the
 *    rendered mesh triangle so narrow streams never suffer terrain "dirt bridges".
 *  - Springs fade in gently at their source; coastal estuaries fade smoothly into sea level.
 *
 * Each river entry: { source: 'lake'|'spring', dest: 'sea'|'lake'|'confluence', points: [[x,z,y,w,slope], ...] }
 */
export function buildRiverSystem(riversData, sampleMeshHeight = null) {
  const positions = [];
  const uvs = [];
  const slopes = [];
  const headFades = [];
  const indices = [];
  let vertOffset = 0;

  function findBankHalfWidth(x, z, dirX, dirZ, maxHalfW, waterY) {
    if (!sampleMeshHeight || maxHalfW <= 2.4) return maxHalfW;
    const steps = 8;
    const minChannelHalf = Math.min(2.4, maxHalfW);
    let openDist = minChannelHalf;
    for (let s = 1; s <= steps; s++) {
      const d = (s / steps) * maxHalfW;
      if (d <= minChannelHalf) {
        openDist = d;
        continue;
      }
      const h = sampleMeshHeight(x + dirX * d, z + dirZ * d);
      // Stop right when we meet a rising riverbank/cliff above the flat water level
      if (h > waterY + 0.35) {
        return Math.min(maxHalfW, openDist + 1.3);
      }
      openDist = d;
    }
    return maxHalfW;
  }

  function smoothArray(arr, passes = 2) {
    const n = arr.length;
    if (n < 3) return arr;
    let cur = arr.slice();
    const tmp = new Float32Array(n);
    for (let p = 0; p < passes; p++) {
      tmp[0] = cur[0];
      tmp[n - 1] = cur[n - 1];
      for (let i = 1; i < n - 1; i++) {
        tmp[i] = cur[i - 1] * 0.25 + cur[i] * 0.5 + cur[i + 1] * 0.25;
      }
      cur = Array.from(tmp);
    }
    return cur;
  }

  for (const river of riversData) {
    const pts = Array.isArray(river) ? river : river.points;
    const isSpring = !Array.isArray(river) && river.source === 'spring';
    const isSeaDest = !Array.isArray(river) && river.dest === 'sea';
    if (!pts || pts.length < 2) continue;

    const n = pts.length;
    const tangentsX = new Float32Array(n);
    const tangentsZ = new Float32Array(n);
    const waterYs = new Float32Array(n);
    const halfLeft = new Array(n);
    const halfRight = new Array(n);

    // Pass 1: tangents + strictly capped flat centerline water levels + bank-confined half-widths
    for (let i = 0; i < n; i++) {
      const [x, z, yRaw, width] = pts[i];
      const prev = pts[Math.max(i - 1, 0)];
      const next = pts[Math.min(i + 1, n - 1)];
      let tx = next[0] - prev[0];
      let tz = next[1] - prev[1];
      const len = Math.hypot(tx, tz) || 1.0;
      tx /= len;
      tz /= len;
      tangentsX[i] = tx;
      tangentsZ[i] = tz;

      const rx = -tz;
      const rz = tx;

      // Physically ground river water surface strictly to the rendered riverbed:
      //  - Floor: bedY + 0.12m (never buried by terrain mesh triangles!)
      //  - Ceiling: bedY + maxWaterDepth (0.28..0.72m — NEVER hovers 3-4ft in the air so you can NEVER walk under it!)
      let y = Math.max(yRaw, 0.04);
      if (sampleMeshHeight) {
        const meshCenterY = sampleMeshHeight(x, z);
        const maxWaterDepth = 0.28 + Math.min(width * 0.014, 0.44);
        const minY = Math.max(meshCenterY + 0.12, 0.04);
        const maxY = Math.max(meshCenterY + maxWaterDepth, 0.04);
        y = Math.max(minY, Math.min(y, maxY));
      }
      waterYs[i] = y;

      const desiredHalfW = width * 0.56;
      halfLeft[i] = findBankHalfWidth(x, z, -rx, -rz, desiredHalfW, y);
      halfRight[i] = findBankHalfWidth(x, z, rx, rz, desiredHalfW, y);
    }

    // Smooth waterYs gently along the channel while preserving ground clearance
    const smoothYs = smoothArray(Array.from(waterYs), 1);
    for (let i = 0; i < n; i++) {
      const [x, z, , width] = pts[i];
      let y = smoothYs[i];
      if (sampleMeshHeight) {
        const bedY = sampleMeshHeight(x, z);
        const maxWaterDepth = 0.30 + Math.min(width * 0.015, 0.46);
        y = Math.max(Math.max(bedY + 0.12, 0.04), Math.min(y, Math.max(bedY + maxWaterDepth, 0.04)));
      }
      waterYs[i] = y;
    }

    // Smooth bank-confined half-widths along the river for organic shorelines
    const smoothL = smoothArray(halfLeft, 2);
    const smoothR = smoothArray(halfRight, 2);

    // Pass 2: build strictly horizontally-flat 4-strip ribbon geometry with ground-tucked edges
    const crossU = [0.0, 0.22, 0.78, 1.0];
    let distAcc = 0.0;

    for (let i = 0; i < n; i++) {
      const [x, z, , , slope] = pts[i];
      if (i > 0) {
        distAcc += Math.hypot(x - pts[i - 1][0], z - pts[i - 1][1]);
      }

      const rx = -tangentsZ[i];
      const rz = tangentsX[i];
      const y = waterYs[i];
      const wL = smoothL[i];
      const wR = smoothR[i];

      // Softly fade in alpine spring starts, cross-fade lake spillway exits, & blend river mouths into the ocean
      const headFade = isSpring ? Math.min(i / 6.0, 1.0) : Math.min((i + 1) / 3.0, 1.0);
      const tailFade = isSeaDest && y < 0.8 ? Math.min((n - 1 - i) / 3.5, 1.0) : 1.0;
      const endFade = Math.min(headFade, tailFade);

      for (let k = 0; k < crossU.length; k++) {
        const u = crossU[k];
        const side = (u - 0.5) * 2.0; // -1..+1
        const offset = side < 0 ? side * wL : side * wR;
        const vx = x + rx * offset;
        const vz = z + rz * offset;
        const isEdge = k === 0 || k === crossU.length - 1;
        // Interior water surface is strictly level at y; outer feathered bank edge tucks cleanly into the ground (zero air gap!)
        let vy = y;
        if (isEdge) {
          vy = y - 0.05;
          if (sampleMeshHeight) {
            const edgeGroundY = sampleMeshHeight(vx, vz);
            if (edgeGroundY < vy) {
              vy = Math.max(edgeGroundY - 0.06, y - 0.65);
            }
          }
        }

        positions.push(vx, vy, vz);
        uvs.push(u, distAcc);
        slopes.push(slope);
        headFades.push(endFade);
      }

      if (i > 0) {
        const rowA = vertOffset + (i - 1) * 4;
        const rowB = vertOffset + i * 4;
        for (let k = 0; k < 3; k++) {
          indices.push(rowA + k, rowB + k, rowA + k + 1);
          indices.push(rowA + k + 1, rowB + k, rowB + k + 1);
        }
      }
    }
    vertOffset += n * 4;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('aSlope', new THREE.Float32BufferAttribute(slopes, 1));
  geo.setAttribute('aHeadFade', new THREE.Float32BufferAttribute(headFades, 1));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    uniforms: {
      uTime: { value: 0.0 },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uFogDensity: { value: 0.00044 },
      uStyleMode: { value: 0 },
    },
    vertexShader: /* glsl */ `
      attribute float aSlope;
      attribute float aHeadFade;
      uniform float uTime;

      varying vec2 vUv;
      varying float vSlope;
      varying float vHeadFade;
      varying vec3 vWorldPos;

      void main() {
        vUv = uv;
        vSlope = aSlope;
        vHeadFade = aHeadFade;
        vec3 pos = position;
        pos.y += sin(uv.y * 0.45 - uTime * 4.5 + uv.x * 5.0) * 0.05 * (1.0 + aSlope * 2.0);
        vec4 wp = modelMatrix * vec4(pos, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform int uStyleMode;

      varying vec2 vUv;
      varying float vSlope;
      varying float vHeadFade;
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
        float speed = 3.0 + vSlope * 10.0;
        vec2 flowUv1 = vec2(vUv.x * 6.5, vUv.y * 0.22 - uTime * speed * 0.30);
        vec2 flowUv2 = vec2(vUv.x * 13.0 + 1.7, vUv.y * 0.48 - uTime * speed * 0.52);

        float n1 = vnoise(flowUv1);
        float n2 = vnoise(flowUv2);
        float ripples = n1 * 0.58 + n2 * 0.42;

        float nx = (vnoise(flowUv1 + vec2(0.18, 0.0)) - n1) * 1.4 + (vnoise(flowUv2 + vec2(0.22, 0.0)) - n2) * 0.8;
        float nz = (vnoise(flowUv1 + vec2(0.0, 0.18)) - n1) * 1.4 + (vnoise(flowUv2 + vec2(0.0, 0.22)) - n2) * 0.8;
        vec3 N = normalize(vec3(nx, 1.0, nz));

        float edgeDist = abs(vUv.x - 0.5) * 2.0;

        vec3 shallowCol = vec3(0.14, 0.48, 0.54);
        vec3 deepCol    = vec3(0.04, 0.20, 0.34);
        vec3 waterCol = mix(deepCol, shallowCol, smoothstep(0.32, 0.90, edgeDist));
        waterCol += vec3(0.05, 0.14, 0.16) * smoothstep(0.55, 0.85, ripples);

        float rapidsIntensity = smoothstep(0.24, 0.52, vSlope);
        float streaks = smoothstep(0.58, 0.82, n1 * 0.5 + n2 * 0.5);
        float whitewater = streaks * rapidsIntensity * 0.85;
        float bankFoam = smoothstep(0.76, 0.96, edgeDist + (n2 - 0.5) * 0.22) * 0.42;
        float foam = clamp(whitewater + bankFoam, 0.0, 0.92);

        vec3 V = normalize(cameraPosition - vWorldPos);
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.6);
        vec3 skyRefl = mix(vec3(0.26, 0.52, 0.78), uFogColor, 0.55);
        waterCol = mix(waterCol, skyRefl, clamp(fresnel * 0.65, 0.0, 0.65));

        vec3 H = normalize(normalize(uSunDir) + V);
        float spec = pow(max(dot(N, H), 0.0), 110.0);
        waterCol += vec3(1.0, 0.97, 0.88) * spec * 1.35;

        vec3 color = mix(waterCol, vec3(0.90, 0.95, 0.99), foam);

        float dist = length(cameraPosition - vWorldPos);
        float density = (uStyleMode == 0) ? uFogDensity : uFogDensity * 0.35;
        float fogFactor = 1.0 - exp(-pow(dist * density, 1.5));
        color = mix(color, uFogColor, clamp(fogFactor, 0.0, 0.90));

        float edgeAlpha = 1.0 - smoothstep(0.70, 1.0, edgeDist);
        float alpha = clamp((0.68 + foam * 0.26 + fresnel * 0.16) * edgeAlpha * vHeadFade, 0.0, 0.94);
        if (alpha < 0.02) discard;

        gl_FragColor = vec4(color, alpha);
      }
    `,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 2;
  return { mesh, material: mat };
}

/**
 * Builds seamless, zero-seam 3D flat alpine & valley lake water sheets sitting
 * at each basin's exact Barnes Priority-Flood spillway elevation.
 *
 * Uses an exact shared-vertex 1x1 cell grid mesh per lake (zero overlapping
 * quads -> zero grid seams AND zero bounding-box bleed between neighboring lakes)
 * while computing continuous per-pixel water depth (`lakeY - terrainY`) from
 * the Float32 height texture so shorelines hug the 3D terrain smoothly!
 */
export function buildLakeSystem(lakesData, gridSize, worldSpan, heightFloatTex, maskTex) {
  const positions = [];
  const levels = [];
  const edgeFades = [];
  const hots = [];
  const indices = [];

  for (const lake of lakesData) {
    if (!lake.cells || lake.cells.length === 0) continue;
    const y = lake.level;

    // Pad lake footprint by 2 cells so organic shorelines (depth -> 0) never hit a hard
    // polygon cutoff, while outer pad vertices carry aEdgeFade -> 0 so spillway outlets
    // cross-fade seamlessly into outgoing rivers (zero sharp geometric cyan edges!).
    const cellDist = new Map();
    const coreSet = new Set();
    for (const [r, c] of lake.cells) {
      const k = r * gridSize + c;
      coreSet.add(k);
      cellDist.set(k, 0);
    }
    // Ring 1 & Ring 2 dilation
    for (let ring = 1; ring <= 2; ring++) {
      const prevKeys = Array.from(cellDist.entries())
        .filter(([, d]) => d === ring - 1)
        .map(([k]) => k);
      for (const k of prevKeys) {
        const r = Math.floor(k / gridSize);
        const c = k % gridSize;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr;
            const nc = c + dc;
            if (nr < 0 || nr >= gridSize || nc < 0 || nc >= gridSize) continue;
            const nk = nr * gridSize + nc;
            if (!cellDist.has(nk)) cellDist.set(nk, ring);
          }
        }
      }
    }

    // Determine smooth fade per grid corner (vr, vc):
    //   1.0 if corner touches any core lake cell (ring 0)
    //   0.55 if corner touches ring 1 (and not ring 0)
    //   0.0 on the outermost rim of ring 2 -> guaranteed zero hard mesh edges!
    const vertFade = (vr, vc) => {
      let minRing = 99;
      for (let dr = -1; dr <= 0; dr++) {
        for (let dc = -1; dc <= 0; dc++) {
          const r = vr + dr;
          const c = vc + dc;
          if (r < 0 || r >= gridSize || c < 0 || c >= gridSize) continue;
          const d = cellDist.get(r * gridSize + c);
          if (d !== undefined && d < minRing) minRing = d;
        }
      }
      // Corner is on the exterior boundary if any of its 4 adjacent cells isn't in cellDist
      let touchesOutside = false;
      for (let dr = -1; dr <= 0; dr++) {
        for (let dc = -1; dc <= 0; dc++) {
          const r = vr + dr;
          const c = vc + dc;
          if (r < 0 || r >= gridSize || c < 0 || c >= gridSize || !cellDist.has(r * gridSize + c)) {
            touchesOutside = true;
          }
        }
      }
      if (touchesOutside) return 0.0;
      if (minRing === 0) return 1.0;
      if (minRing === 1) return 0.65;
      return 0.0;
    };

    // Deduplicate grid-corner vertices (vr, vc) within this lake so adjacent cells share
    // exact edges (zero double-blended alpha seams, strictly disjoint from other lakes).
    const vertMap = new Map();
    const getVert = (vr, vc) => {
      const key = vr * (gridSize + 2) + vc;
      let idx = vertMap.get(key);
      if (idx === undefined) {
        idx = positions.length / 3;
        const wx = ((vc - 0.5) / (gridSize - 1) - 0.5) * worldSpan;
        const wz = ((vr - 0.5) / (gridSize - 1) - 0.5) * worldSpan;
        positions.push(wx, y, wz);
        levels.push(y);
        edgeFades.push(vertFade(vr, vc));
        hots.push(lake.hotSpring ? 1 : 0);
        vertMap.set(key, idx);
      }
      return idx;
    };

    for (const k of cellDist.keys()) {
      const r = Math.floor(k / gridSize);
      const c = k % gridSize;
      const v00 = getVert(r, c);
      const v10 = getVert(r, c + 1);
      const v01 = getVert(r + 1, c);
      const v11 = getVert(r + 1, c + 1);
      indices.push(v00, v01, v10, v10, v01, v11);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('aLevel', new THREE.Float32BufferAttribute(levels, 1));
  geo.setAttribute('aEdgeFade', new THREE.Float32BufferAttribute(edgeFades, 1));
  // 1 on a volcano's hot spring pool (see fill_hot_springs in experiments/volcano.py)
  geo.setAttribute('aHot', new THREE.Float32BufferAttribute(hots, 1));
  geo.setIndex(indices);

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uHeightTex: { value: heightFloatTex },
      uMasks: { value: maskTex },
      // Span and world-XZ centre of the height / mask textures (the seam atlas is larger than a tile)
      uWorldSpan: { value: worldSpan },
      uTexOrigin: { value: new THREE.Vector2(0, 0) },
      uVScale: { value: 1.0 },
      uTime: { value: 0.0 },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uFogDensity: { value: 0.00044 },
      uStyleMode: { value: 0 },
    },
    vertexShader: /* glsl */ `
      attribute float aLevel;
      attribute float aEdgeFade;
      attribute float aHot;
      varying float vLevel;
      varying float vEdgeFade;
      varying float vHot;
      varying vec3 vWorldPos;

      void main() {
        vLevel = aLevel;
        vEdgeFade = aEdgeFade;
        vHot = aHot;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uHeightTex;
      uniform sampler2D uMasks;
      uniform float uWorldSpan;
      uniform vec2 uTexOrigin;
      uniform float uVScale;
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform int uStyleMode;

      varying float vLevel;
      varying float vEdgeFade;
      varying float vHot;
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
        float hot = step(0.5, vHot);
        // Compute exact continuous water depth from Float32 heightmap
        vec2 uv = (vWorldPos.xz - uTexOrigin) / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y;
        float bedY = texture2D(uHeightTex, uv).r * uVScale;
        float waterY = vLevel * uVScale;
        float depth = waterY - bedY;

        // Discard pixels outside the basin where terrain pokes above the flat lake surface
        if (depth <= 0.0) discard;

        // Gentle wind-driven alpine lake ripples
        vec2 p1 = vWorldPos.xz * 0.24 + vec2(uTime * 0.35, uTime * 0.22);
        vec2 p2 = vWorldPos.xz * 0.46 - vec2(uTime * 0.28, -uTime * 0.30);
        float w1 = vnoise(p1);
        float w2 = vnoise(p2);
        vec3 N = normalize(vec3((w1 - 0.5) * 0.16 + (w2 - 0.5) * 0.10, 1.0, (w2 - 0.5) * 0.16 - (w1 - 0.5) * 0.10));

        // Crystal emerald-turquoise mountain shallows -> deep alpine sapphire center
        vec3 shallowCol = vec3(0.14, 0.54, 0.58);
        vec3 deepCol    = vec3(0.04, 0.20, 0.36);
        vec3 waterCol = mix(shallowCol, deepCol, smoothstep(0.25, 5.5, depth));

        // Delicate organic shoreline foam ring
        float shoreFoam = (1.0 - smoothstep(0.0, 0.65, depth)) * (0.45 + 0.35 * sin(uTime * 2.2 + w1 * 5.0));
        waterCol = mix(waterCol, vec3(0.90, 0.96, 1.0), shoreFoam * 0.55 * (1.0 - hot));

        // A hot spring: rings of heat-loving microbes, each colour at the depth (and so the
        // temperature) it can stand, around water too hot for anything and so a clear deep blue
        if (hot > 0.5) {
          float d = depth + (vnoise(vWorldPos.xz * 0.11) - 0.5) * 1.1;
          vec3 c = vec3(0.62, 0.24, 0.06);
          c = mix(c, vec3(0.93, 0.52, 0.08), smoothstep(0.1, 0.7, d));
          c = mix(c, vec3(0.95, 0.83, 0.22), smoothstep(0.7, 1.6, d));
          c = mix(c, vec3(0.22, 0.74, 0.44), smoothstep(1.6, 2.8, d));
          c = mix(c, vec3(0.04, 0.68, 0.80), smoothstep(2.8, 4.6, d));
          c = mix(c, vec3(0.02, 0.26, 0.66), smoothstep(5.0, 16.0, d));
          // Steam lying on the surface, drifting
          float wisps = vnoise(vWorldPos.xz * 0.05 + vec2(uTime * 0.05, -uTime * 0.03)) * vnoise(vWorldPos.xz * 0.13 - uTime * 0.04);
          waterCol = mix(c, vec3(0.94, 0.96, 0.96), smoothstep(0.22, 0.6, wisps) * 0.4);
        }

        // Mirror-like Fresnel sky reflection + crisp sun glints
        vec3 V = normalize(cameraPosition - vWorldPos);
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.2);
        vec3 skyRefl = mix(vec3(0.26, 0.54, 0.80), uFogColor, 0.55);
        waterCol = mix(waterCol, skyRefl, clamp(fresnel * 0.68, 0.0, 0.68) * (1.0 - 0.6 * hot));

        vec3 H = normalize(normalize(uSunDir) + V);
        float spec = pow(max(dot(N, H), 0.0), 120.0);
        waterCol += vec3(1.0, 0.97, 0.88) * spec * 1.4;

        float dist = length(cameraPosition - vWorldPos);
        float density = (uStyleMode == 0) ? uFogDensity : uFogDensity * 0.35;
        float fogFactor = 1.0 - exp(-pow(dist * density, 1.5));
        waterCol = mix(waterCol, uFogColor, clamp(fogFactor, 0.0, 0.90));

        // Feather smoothly to alpha=0 right at the waterline AND outer spillway pad rim!
        float rimFade = smoothstep(0.0, 0.55, vEdgeFade);
        float alpha = clamp(smoothstep(0.0, 0.40, depth) * rimFade * (0.68 + smoothstep(0.0, 2.5, depth) * 0.28 + fresnel * 0.12), 0.0, 0.95);
        // The mineral colours are in the water itself: a hot pool hides its bed
        alpha = mix(alpha, clamp(smoothstep(0.0, 0.25, depth) * rimFade * 0.96, 0.0, 0.96), hot);
        // A crater rim is a thin crest with lower ground just outside it: keep the pool to the
        // lake mask, or its sheet shows on the outer slope
        if (hot > 0.5) alpha *= smoothstep(0.12, 0.45, texture2D(uMasks, uv).g);
        if (alpha < 0.015) discard;
        gl_FragColor = vec4(waterCol, alpha);
      }
    `,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 2;
  return { mesh, material: mat };
}
