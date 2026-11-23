import * as THREE from 'three';

/**
 * Builds real 3D flowing river ribbon meshes along physically solved
 * lake-to-sea and alpine-spring-to-sea drainage splines.
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

  for (const river of riversData) {
    const pts = Array.isArray(river) ? river : river.points;
    const isSpring = !Array.isArray(river) && river.source === 'spring';
    if (!pts || pts.length < 2) continue;

    const crossU = [0.0, 0.22, 0.78, 1.0];
    let distAcc = 0.0;

    for (let i = 0; i < pts.length; i++) {
      const [x, z, yRaw, width, slope] = pts[i];
      if (i > 0) {
        distAcc += Math.hypot(x - pts[i - 1][0], z - pts[i - 1][1]);
      }

      const prev = pts[Math.max(i - 1, 0)];
      const next = pts[Math.min(i + 1, pts.length - 1)];
      let tx = next[0] - prev[0];
      let tz = next[1] - prev[1];
      const len = Math.hypot(tx, tz) || 1.0;
      tx /= len;
      tz /= len;

      const rx = -tz;
      const rz = tx;
      const halfW = width * 0.56;

      // Guarantee centerline sits cleanly above the actual rendered terrain triangle (zero dirt bridges!)
      let y = yRaw;
      if (sampleMeshHeight) {
        const meshCenterY = sampleMeshHeight(x, z);
        if (meshCenterY + 0.14 > y) {
          y = meshCenterY + 0.14;
        }
      }

      // Softly fade in the first few meters of an alpine spring trickle
      const headFade = isSpring ? Math.min(i / 6.0, 1.0) : 1.0;

      for (let k = 0; k < crossU.length; k++) {
        const u = crossU[k];
        const side = (u - 0.5) * 2.0;
        const vx = x + rx * side * halfW;
        const vz = z + rz * side * halfW;
        const isEdge = k === 0 || k === crossU.length - 1;
        let vy = y + (isEdge ? -0.05 : 0.0);
        // For interior water vertices, ensure rendered mesh never pokes through the water surface
        if (!isEdge && sampleMeshHeight) {
          const mY = sampleMeshHeight(vx, vz);
          if (mY + 0.10 > vy) vy = mY + 0.10;
        }
        positions.push(vx, vy, vz);
        uvs.push(u, distAcc);
        slopes.push(slope);
        headFades.push(headFade);
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
    vertOffset += pts.length * 4;
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
        pos.y += sin(uv.y * 0.45 - uTime * 4.5 + uv.x * 5.0) * 0.07 * (1.0 + aSlope * 2.5);
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
  const indices = [];

  for (const lake of lakesData) {
    if (!lake.cells || lake.cells.length === 0) continue;
    const y = lake.level;
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
        vertMap.set(key, idx);
      }
      return idx;
    };

    for (const [r, c] of lake.cells) {
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
  geo.setIndex(indices);

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uHeightTex: { value: heightFloatTex },
      uMasks: { value: maskTex },
      uWorldSpan: { value: worldSpan },
      uVScale: { value: 1.0 },
      uTime: { value: 0.0 },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uFogDensity: { value: 0.00044 },
      uStyleMode: { value: 0 },
    },
    vertexShader: /* glsl */ `
      attribute float aLevel;
      varying float vLevel;
      varying vec3 vWorldPos;

      void main() {
        vLevel = aLevel;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uHeightTex;
      uniform sampler2D uMasks;
      uniform float uWorldSpan;
      uniform float uVScale;
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform int uStyleMode;

      varying float vLevel;
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
        // Compute exact continuous water depth from Float32 heightmap
        vec2 uv = vWorldPos.xz / uWorldSpan + 0.5;
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
        waterCol = mix(waterCol, vec3(0.90, 0.96, 1.0), shoreFoam * 0.55);

        // Mirror-like Fresnel sky reflection + crisp sun glints
        vec3 V = normalize(cameraPosition - vWorldPos);
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.2);
        vec3 skyRefl = mix(vec3(0.26, 0.54, 0.80), uFogColor, 0.55);
        waterCol = mix(waterCol, skyRefl, clamp(fresnel * 0.68, 0.0, 0.68));

        vec3 H = normalize(normalize(uSunDir) + V);
        float spec = pow(max(dot(N, H), 0.0), 120.0);
        waterCol += vec3(1.0, 0.97, 0.88) * spec * 1.4;

        float dist = length(cameraPosition - vWorldPos);
        float density = (uStyleMode == 0) ? uFogDensity : uFogDensity * 0.35;
        float fogFactor = 1.0 - exp(-pow(dist * density, 1.5));
        waterCol = mix(waterCol, uFogColor, clamp(fogFactor, 0.0, 0.90));

        // Feather smoothly to alpha=0 right at the waterline
        float alpha = clamp(smoothstep(0.0, 0.35, depth) * (0.68 + smoothstep(0.0, 2.5, depth) * 0.28 + fresnel * 0.12), 0.0, 0.95);
        gl_FragColor = vec4(waterCol, alpha);
      }
    `,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 2;
  return { mesh, material: mat };
}
