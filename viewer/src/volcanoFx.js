import * as THREE from 'three';

/**
 * What an active volcano does in the explorer: a pool of lava on the crater floor, a glow
 * above it that strengthens as the sun goes down, and a column of smoke that leans downwind.
 *
 * Everything is animated in the shaders from a clock, so there is no per-frame work on the
 * CPU beyond setting uniforms. Positions are in unscaled viewer metres; the terrain's
 * vertical-scale slider is applied in the shaders.
 */

// Crater radius as a share of the cone's radius (matches VOLCANO_KINDS in tectonics.js)
const CRATER = { strato: 0.1, shield: 0.15, cinder: 0.26 };
const PUFFS = 64;

const SMOKE_VERT = /* glsl */ `
  attribute vec3 aBase;   // crater floor
  attribute float aSeed;
  attribute float aSize;  // crater radius (m)
  uniform float uTime;
  uniform float uVScale;
  uniform float uPixels;  // viewport height / (2 tan(fov / 2))
  uniform vec2 uWind;
  varying float vAge;
  varying float vSeed;
  void main() {
    float speed = 0.028 + 0.012 * fract(aSeed * 7.31);
    float age = fract(uTime * speed + aSeed);
    vAge = age;
    vSeed = aSeed;
    float rise = aSize * 13.0 * pow(age, 0.82);
    float ang = aSeed * 39.7 + uTime * 0.07;
    float spread = aSize * (0.25 + 2.1 * age);
    vec3 p = vec3(aBase.x, aBase.y * uVScale, aBase.z);
    p.y += rise + aSize * 0.4;
    p.xz += vec2(cos(ang), sin(ang)) * spread * fract(aSeed * 13.7);
    // The column bends over as it climbs into the wind
    p.xz += uWind * aSize * 9.0 * age * age;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(aSize * (1.3 + 4.6 * age) * uPixels / max(-mv.z, 1.0), 420.0);
  }
`;

const SMOKE_FRAG = /* glsl */ `
  uniform float uNight;
  uniform vec3 uSunColor;
  varying float vAge;
  varying float vSeed;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c) * 2.0;
    float soft = 1.0 - smoothstep(0.25, 1.0, d);
    if (soft < 0.01) discard;
    // Ash-dark where it leaves the vent, paling as it thins out
    vec3 col = mix(vec3(0.16, 0.15, 0.15), vec3(0.72, 0.71, 0.70), smoothstep(0.05, 0.7, vAge));
    col *= mix(0.86, 1.08, fract(vSeed * 5.3));
    // Lit from above by the sun, from below by the crater
    col *= mix(0.42, 1.0, 1.0 - uNight) * mix(vec3(1.0), uSunColor, 0.35);
    col += vec3(1.0, 0.36, 0.08) * (1.0 - smoothstep(0.0, 0.2, vAge)) * (0.25 + 0.75 * uNight) * (0.6 - c.y);
    float alpha = soft * smoothstep(0.0, 0.06, vAge) * pow(1.0 - vAge, 1.3) * 0.58;
    gl_FragColor = vec4(col, alpha);
  }
`;

const LAVA_VERT = /* glsl */ `
  attribute vec3 aBase;
  attribute float aSize;
  uniform float uVScale;
  varying vec2 vLocal;
  varying vec2 vWorld;
  void main() {
    vLocal = position.xz;
    vec3 p = vec3(aBase.x + position.x * aSize, aBase.y * uVScale + 0.6, aBase.z + position.z * aSize);
    vWorld = p.xz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const LAVA_FRAG = /* glsl */ `
  uniform float uTime;
  varying vec2 vLocal;
  varying vec2 vWorld;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
  }
  void main() {
    float r = length(vLocal);
    if (r > 1.0) discard;
    // Cooled crust in slowly drifting plates, split by bright cracks
    vec2 q = vWorld * 0.16 + vec2(uTime * 0.012, -uTime * 0.008);
    float plates = vnoise(q) * 0.6 + vnoise(q * 2.3 + 4.1) * 0.4;
    float crack = 1.0 - smoothstep(0.0, 0.09, abs(plates - 0.5));
    float pulse = 0.75 + 0.25 * sin(uTime * 0.7 + plates * 9.0);
    vec3 crust = vec3(0.10, 0.03, 0.02) + vec3(0.25, 0.05, 0.0) * plates;
    vec3 molten = mix(vec3(1.0, 0.34, 0.04), vec3(1.0, 0.86, 0.42), crack * pulse);
    vec3 col = mix(crust, molten * 1.5, max(crack * pulse, (1.0 - smoothstep(0.0, 0.5, r)) * 0.35));
    gl_FragColor = vec4(col, 1.0 - smoothstep(0.8, 1.0, r));
  }
`;

const GLOW_VERT = /* glsl */ `
  attribute vec3 aBase;
  attribute float aSize;
  uniform float uVScale;
  uniform float uPixels;
  void main() {
    vec3 p = vec3(aBase.x, aBase.y * uVScale + aSize * 1.2, aBase.z);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(aSize * 9.0 * uPixels / max(-mv.z, 1.0), 600.0);
  }
`;

const GLOW_FRAG = /* glsl */ `
  uniform float uTime;
  uniform float uNight;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float g = pow(max(0.0, 1.0 - d), 2.2);
    float flicker = 0.85 + 0.15 * sin(uTime * 1.9) * sin(uTime * 0.83);
    gl_FragColor = vec4(vec3(1.0, 0.42, 0.10) * g * flicker * (0.16 + 0.6 * uNight), 1.0);
  }
`;

/**
 * @param {THREE.Scene} scene
 * @returns {{
 *   set: (volcanoes: {id: number, kind: string, x: number, z: number, y: number, radiusM: number}[]) => void,
 *   update: (t: number, camera: THREE.PerspectiveCamera, viewportHeight: number, sunDir: THREE.Vector3, vScale: number) => void,
 *   dispose: () => void,
 * }} `set` takes the active volcanoes in view, with `y` the unscaled height of the crater floor.
 */
export function createVolcanoFx(scene) {
  const uniforms = {
    uTime: { value: 0 },
    uVScale: { value: 1 },
    uPixels: { value: 600 },
    uNight: { value: 0 },
    uWind: { value: new THREE.Vector2(0.8, 0.35) },
    uSunColor: { value: new THREE.Color(1, 0.96, 0.9) },
  };
  let group = null;
  let signature = '';

  function clear() {
    if (!group) return;
    scene.remove(group);
    group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    group = null;
  }

  function set(volcanoes) {
    const sig = volcanoes.map((v) => `${v.id}:${v.x.toFixed(0)},${v.z.toFixed(0)},${v.y.toFixed(1)}`).join('|');
    if (sig === signature) return;
    signature = sig;
    clear();
    if (!volcanoes.length) return;
    group = new THREE.Group();
    const n = volcanoes.length;
    const size = (v) => Math.max(8, v.radiusM * (CRATER[v.kind] ?? 0.12));

    // Smoke: PUFFS points per volcano, each at its own point in the same rising cycle
    const sBase = new Float32Array(n * PUFFS * 3);
    const sSeed = new Float32Array(n * PUFFS);
    const sSize = new Float32Array(n * PUFFS);
    volcanoes.forEach((v, i) => {
      for (let k = 0; k < PUFFS; k++) {
        const o = i * PUFFS + k;
        sBase.set([v.x, v.y, v.z], o * 3);
        sSeed[o] = (k + ((v.id * 0.6180339) % 1)) / PUFFS + ((k * 0.754877) % 1) * 0.01;
        sSize[o] = size(v);
      }
    });
    const smokeGeo = new THREE.BufferGeometry();
    smokeGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * PUFFS * 3), 3));
    smokeGeo.setAttribute('aBase', new THREE.BufferAttribute(sBase, 3));
    smokeGeo.setAttribute('aSeed', new THREE.BufferAttribute(sSeed, 1));
    smokeGeo.setAttribute('aSize', new THREE.BufferAttribute(sSize, 1));
    const smoke = new THREE.Points(
      smokeGeo,
      new THREE.ShaderMaterial({ uniforms, vertexShader: SMOKE_VERT, fragmentShader: SMOKE_FRAG, transparent: true, depthWrite: false })
    );
    smoke.frustumCulled = false;
    smoke.renderOrder = 6;

    // One vertex per volcano for the glow, and a disc each for the lava
    const base = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    volcanoes.forEach((v, i) => {
      base.set([v.x, v.y, v.z], i * 3);
      sizes[i] = size(v);
    });
    const glowGeo = new THREE.BufferGeometry();
    glowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    glowGeo.setAttribute('aBase', new THREE.BufferAttribute(base, 3));
    glowGeo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    const glow = new THREE.Points(
      glowGeo,
      new THREE.ShaderMaterial({
        uniforms,
        vertexShader: GLOW_VERT,
        fragmentShader: GLOW_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    glow.frustumCulled = false;
    glow.renderOrder = 5;

    const disc = new THREE.CircleGeometry(1, 28).rotateX(-Math.PI / 2);
    const lavaGeo = new THREE.InstancedBufferGeometry();
    lavaGeo.index = disc.index;
    lavaGeo.setAttribute('position', disc.getAttribute('position'));
    lavaGeo.setAttribute('aBase', new THREE.InstancedBufferAttribute(base, 3));
    lavaGeo.setAttribute('aSize', new THREE.InstancedBufferAttribute(sizes.map((s) => s * 0.8), 1));
    lavaGeo.instanceCount = n;
    const lava = new THREE.Mesh(
      lavaGeo,
      new THREE.ShaderMaterial({ uniforms, vertexShader: LAVA_VERT, fragmentShader: LAVA_FRAG, transparent: true })
    );
    lava.frustumCulled = false;

    group.add(lava, glow, smoke);
    scene.add(group);
  }

  function update(t, camera, viewportHeight, sunDir, vScale) {
    uniforms.uTime.value = t;
    uniforms.uVScale.value = vScale;
    uniforms.uPixels.value = viewportHeight / (2 * Math.tan((camera.fov * Math.PI) / 360));
    const up = Math.max(0, Math.min(1, (sunDir.y - 0.05) / 0.45));
    uniforms.uNight.value = 1 - up * up * (3 - 2 * up);
    uniforms.uSunColor.value.setRGB(1, 0.72 + 0.24 * up, 0.46 + 0.43 * up);
  }

  return {
    set,
    update,
    dispose() {
      clear();
      signature = '';
    },
  };
}
