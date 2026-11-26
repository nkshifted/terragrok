import * as THREE from 'three';

/**
 * GPU-Instanced 3D Wind-Swaying Grass Field (45,000+ curved blades in 1 draw call).
 *
 * Uses a seamless toroidal ring centered on `uFocusPos` (the player/camera) so
 * grass follows you anywhere across the 3 km world with ZERO per-frame CPU work.
 * Vertex shader reads exact Float32 elevation, terrain slope, river/lake/moisture
 * masks, matches base blade color to ground albedo (eliminating pop-in seams!),
 * animates rolling wind gusts, and parts blades around the explorer's feet.
 */
export function createGrassField(heightFloatTex, maskTex, worldSpan, maxElev, bladeCount = 48000, ringSize = 155.0) {
  // Curved, tapered 3-segment grass blade geometry (7 vertices, 5 triangles)
  const baseGeo = new THREE.BufferGeometry();
  const w0 = 0.14;
  const w1 = 0.10;
  const w2 = 0.05;
  const verts = new Float32Array([
    -w0, 0.0, 0.0,   w0, 0.0, 0.0,
    -w1, 0.36, 0.03, w1, 0.36, 0.03,
    -w2, 0.72, 0.11, w2, 0.72, 0.11,
     0.0, 1.08, 0.24,
  ]);
  const tipFrac = new Float32Array([0.0, 0.0, 0.35, 0.35, 0.7, 0.7, 1.0]);
  const indices = [
    0, 1, 2,  1, 3, 2,
    2, 3, 4,  3, 5, 4,
    4, 5, 6,
  ];
  baseGeo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  baseGeo.setAttribute('aTip', new THREE.BufferAttribute(tipFrac, 1));
  baseGeo.setIndex(indices);

  // Per-instance random offsets inside the [0, ringSize) tile + angle + height jitter
  const instGeo = new THREE.InstancedBufferGeometry();
  instGeo.index = baseGeo.index;
  instGeo.attributes.position = baseGeo.attributes.position;
  instGeo.attributes.aTip = baseGeo.attributes.aTip;

  const offsets = new Float32Array(bladeCount * 4); // ox, oz, rotY, scaleJitter
  let s = 9001;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  for (let i = 0; i < bladeCount; i++) {
    offsets[i * 4 + 0] = rand() * ringSize;
    offsets[i * 4 + 1] = rand() * ringSize;
    offsets[i * 4 + 2] = rand() * Math.PI * 2.0;
    offsets[i * 4 + 3] = 0.65 + rand() * 0.95;
  }
  instGeo.setAttribute('aInst', new THREE.InstancedBufferAttribute(offsets, 4));
  instGeo.instanceCount = bladeCount;

  const mat = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    uniforms: {
      uHeightTex: { value: heightFloatTex },
      uMasks: { value: maskTex },
      uWorldSpan: { value: worldSpan },
      uTexOrigin: { value: new THREE.Vector2(0, 0) },
      uMaxElev: { value: maxElev },
      uVScale: { value: 1.0 },
      uRingSize: { value: ringSize },
      uFocusPos: { value: new THREE.Vector3() },
      uPlayerPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uMistColor: { value: new THREE.Color('#d2e4f0') },
      uFogDensity: { value: 0.00044 },
      uTime: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      attribute float aTip;
      attribute vec4 aInst; // ox, oz, rotY, heightScale

      uniform sampler2D uHeightTex;
      uniform sampler2D uMasks;
      uniform float uWorldSpan;
      uniform vec2 uTexOrigin;
      uniform float uMaxElev;
      uniform float uVScale;
      uniform float uRingSize;
      uniform vec3 uFocusPos;
      uniform vec3 uPlayerPos;
      uniform float uTime;

      varying float vTip;
      varying float vMoisture;
      varying float vHNorm;
      varying float vFade;
      varying float vWind;
      varying vec3 vWorldPos;
      varying vec3 vTerrainNormal;

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

      float sampleH(vec2 xz) {
        vec2 uv = (xz - uTexOrigin) / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y;
        return texture2D(uHeightTex, uv).r * uVScale;
      }

      void main() {
        // Wrap blade world XZ seamlessly inside ring centered on uFocusPos
        float halfR = uRingSize * 0.5;
        vec2 rel = mod(aInst.xy - uFocusPos.xz + halfR, uRingSize) - halfR;
        vec2 wxz = uFocusPos.xz + rel;

        // World UV check
        vec2 uv = (wxz - uTexOrigin) / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y;
        bool inBounds = uv.x > 0.01 && uv.x < 0.99 && uv.y > 0.01 && uv.y < 0.99;

        // Smooth circular fade near ring boundary so blades never pop
        float distFromFocus = length(rel);
        float ringFade = 1.0 - smoothstep(halfR * 0.65, halfR * 0.98, distFromFocus);

        float elevRaw = inBounds ? texture2D(uHeightTex, uv).r : -20.0;
        float elev = elevRaw * uVScale;
        vec4 m = inBounds ? texture2D(uMasks, uv) : vec4(0.0);
        float riverWet = m.r;
        float lake = m.g;
        float moisture = m.a;

        // Estimate local terrain normal & slope from 4 height taps
        float eps = 3.5;
        float hL = sampleH(wxz - vec2(eps, 0.0));
        float hR = sampleH(wxz + vec2(eps, 0.0));
        float hD = sampleH(wxz - vec2(0.0, eps));
        float hU = sampleH(wxz + vec2(0.0, eps));
        vec3 tNorm = normalize(vec3(hL - hR, 2.0 * eps, hD - hU));
        float slope = 1.0 - clamp(tNorm.y, 0.0, 1.0);

        float hNorm = clamp(elevRaw / max(uMaxElev, 1.0), 0.0, 1.0);

        // Grass validity mask: above beach, below alpine snow/rock, off cliffs, out of rivers/lakes,
        // AND strictly gated by biome moisture (0 grass in deserts & frozen glaciers!)
        float valid = inBounds ? 1.0 : 0.0;
        valid *= smoothstep(3.8, 8.2, elevRaw);
        valid *= smoothstep(0.18, 0.42, moisture);
        valid *= (1.0 - smoothstep(0.48, 0.60, hNorm));
        valid *= (1.0 - smoothstep(0.09, 0.16, slope));
        valid *= (1.0 - smoothstep(0.32, 0.55, riverWet));
        valid *= (1.0 - smoothstep(0.20, 0.45, lake));

        // Taller lush reeds right near riverbanks, shorter dry grass in steppe
        float reedBoost = smoothstep(0.06, 0.28, riverWet) * (1.0 - smoothstep(0.30, 0.50, riverWet)) * 0.85;
        float bladeScale = aInst.w * (0.72 + 0.55 * moisture + reedBoost) * valid * ringFade;

        // Rotate blade randomly around Y
        float c = cos(aInst.z);
        float s = sin(aInst.z);
        vec3 localPos = position;
        localPos.x *= (0.9 + 0.4 * moisture);
        localPos.y *= bladeScale;
        localPos.z *= bladeScale;
        vec3 rotPos = vec3(
          localPos.x * c - localPos.z * s,
          localPos.y,
          localPos.x * s + localPos.z * c
        );

        // Rolling wind gust waves sweeping across the landscape
        float windWave = vnoise(wxz * 0.055 + vec2(uTime * 0.85, uTime * 0.42));
        float gust = (windWave - 0.35) * 1.15 + sin(uTime * 2.4 + wxz.x * 0.18 + wxz.y * 0.12) * 0.22;
        float bend = aTip * aTip * bladeScale;
        rotPos.x += gust * bend * 0.65;
        rotPos.z += gust * bend * 0.35;
        rotPos.y -= abs(gust) * bend * 0.16;

        // Part grass softly around the explorer's feet!
        vec2 toPlayer = wxz - uPlayerPos.xz;
        float pDist = length(toPlayer);
        if (pDist < 2.6 && pDist > 0.01) {
          float push = (1.0 - pDist / 2.6) * aTip * bladeScale * 1.15;
          vec2 pushDir = toPlayer / pDist;
          rotPos.x += pushDir.x * push;
          rotPos.z += pushDir.y * push;
          rotPos.y -= push * 0.45;
        }

        vec3 worldPos = vec3(wxz.x, elev - 0.08, wxz.y) + rotPos;

        vTip = aTip;
        vMoisture = moisture;
        vHNorm = hNorm;
        vFade = valid * ringFade;
        vWind = windWave;
        vWorldPos = worldPos;
        vTerrainNormal = tNorm;

        gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform vec3 uFogColor;
      uniform vec3 uMistColor;
      uniform float uFogDensity;

      varying float vTip;
      varying float vMoisture;
      varying float vHNorm;
      varying float vFade;
      varying float vWind;
      varying vec3 vWorldPos;
      varying vec3 vTerrainNormal;

      void main() {
        if (vFade < 0.03) discard;

        // Match exact terrain ground albedo at base -> warm sunlit tips!
        vec3 aridBase  = vec3(0.60, 0.54, 0.32);
        vec3 meadowBase = vec3(0.24, 0.44, 0.18);
        vec3 lushBase  = vec3(0.14, 0.34, 0.15);

        vec3 baseCol = mix(aridBase, meadowBase, smoothstep(0.22, 0.56, vMoisture));
        baseCol = mix(baseCol, lushBase, smoothstep(0.62, 0.90, vMoisture));
        baseCol = mix(baseCol, vec3(0.36, 0.39, 0.26), smoothstep(0.42, 0.70, vHNorm));

        vec3 tipCol = mix(vec3(0.84, 0.78, 0.44), vec3(0.50, 0.76, 0.30), smoothstep(0.22, 0.65, vMoisture));
        // Sunlit wind-gust shimmer highlights
        tipCol += vec3(0.12, 0.14, 0.05) * smoothstep(0.55, 0.85, vWind);

        vec3 albedo = mix(baseCol * 0.72, tipCol, pow(vTip, 0.78));

        // Soft terrain-aligned diffuse + sunlit blade translucency
        vec3 sunDir = normalize(uSunDir);
        float NdotL = max(dot(normalize(vTerrainNormal), sunDir), 0.0);
        vec3 sunColor = mix(vec3(1.0, 0.74, 0.48), vec3(1.0, 0.96, 0.89), smoothstep(0.10, 0.52, sunDir.y));
        vec3 skyAmb = vec3(0.36, 0.50, 0.68);

        // Back-lit grass translucency when looking toward sun
        vec3 V = normalize(cameraPosition - vWorldPos);
        float backLight = pow(max(dot(-V, sunDir), 0.0), 3.0) * vTip * 0.45;

        vec3 color = albedo * (skyAmb * (0.45 + 0.35 * vTip) + sunColor * (NdotL * 1.15 + backLight));

        // Match terrain valley mist & aerial perspective
        float dist = length(cameraPosition - vWorldPos);
        float heightFactor = exp(-max(vWorldPos.y, 0.0) * 0.014);
        float valleyMist = (1.0 - exp(-dist * 0.00065)) * heightFactor * 0.48;
        color = mix(color, uMistColor, clamp(valleyMist, 0.0, 0.55));

        float aerial = 1.0 - exp(-pow(dist * uFogDensity, 1.5));
        color = mix(color, uFogColor, clamp(aerial, 0.0, 0.90));

        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });

  const mesh = new THREE.Mesh(instGeo, mat);
  mesh.frustumCulled = false; // ring dynamically follows camera on GPU
  return { mesh, material: mat };
}
