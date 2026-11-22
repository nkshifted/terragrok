import * as THREE from 'three';

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
      uTempRange: { value: new THREE.Vector2(tempRange[0], tempRange[1]) },
      uPrecipRange: { value: new THREE.Vector2(precipRange[0], precipRange[1]) },
      uWorldSpan: { value: worldSpan },
      uGridSize: { value: gridSize },
      uSunDir: { value: new THREE.Vector3(0.65, 0.45, 0.45).normalize() },
      uFogColor: { value: new THREE.Color('#a8d0ea') },
      uMistColor: { value: new THREE.Color('#d2e4f0') },
      uFogDensity: { value: 0.00044 },
      uMaxElev: { value: maxElev },
      uVScale: { value: 1.0 },
      uStyleMode: { value: 0 },
      uTime: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vElevRaw;

      void main() {
        vUv = uv;
        vElevRaw = position.y;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uMasks;
      uniform sampler2D uHeightTex;
      uniform sampler2D uClimate;
      uniform int uHasClimate;
      uniform vec2 uTempRange;
      uniform vec2 uPrecipRange;
      uniform float uWorldSpan;
      uniform float uGridSize;
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

      // Sample exact terrain height (meters * vScale) from float height texture
      float sampleHeightAtWorld(vec2 xz) {
        vec2 uv = xz / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y; // match PlaneGeometry flipY
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return -50.0;
        return texture2D(uHeightTex, uv).r * uVScale;
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

        vec3 nSmooth = normalize(vNormalW);
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

        // Biome ground blend driven by temperature + precipitation + riparian wetness
        float effMoist = clamp(mix(moisture, precipNorm, uHasClimate == 1 ? 0.55 : 0.0) + riverWet * 0.45, 0.0, 1.0);
        vec3 warmGround = mix(aridSteppe, warmScrub, smoothstep(0.18, 0.48, effMoist));
        vec3 lushGround = mix(meadow, lushMoss, smoothstep(0.52, 0.88, effMoist));
        vec3 grassCol = mix(warmGround, lushGround, smoothstep(0.30, 0.65, effMoist));
        if (uHasClimate == 1) {
          // Warm dry lowlands stay sun-baked golden steppe; cool slopes shift boreal -> tundra
          float warmDry = smoothstep(9.0, 17.0, tempC) * (1.0 - smoothstep(0.18, 0.50, precipNorm));
          grassCol = mix(grassCol, aridSteppe * 1.04, warmDry * 0.72);
          float coolBoreal = (1.0 - smoothstep(2.0, 9.5, tempC)) * smoothstep(0.25, 0.65, effMoist);
          grassCol = mix(grassCol, borealPine, coolBoreal * 0.62);
          float coldTundra = 1.0 - smoothstep(0.5, 4.5, tempC);
          grassCol = mix(grassCol, alpineTund, coldTundra * 0.75);
        } else {
          grassCol = mix(grassCol, alpineTund, smoothstep(0.42, 0.72, hNorm));
        }
        // Natural soil & clover mottling at 3 scales (breaks up flat green)
        grassCol = mix(grassCol * 0.80, grassCol * 1.15, macroNoise * 0.55 + midNoise * 0.30 + fineNoise * 0.15);

        // Beach transition (wet shore -> dry sand -> grass)
        vec3 sandCol = mix(wetSand, drySand, smoothstep(0.3, 2.8, elev));
        vec3 albedo = mix(seabed, sandCol, smoothstep(-5.0, 0.8, elev));
        albedo = mix(albedo, grassCol, smoothstep(2.4, 8.5, elev));

        // Detailed stratified rock on slopes & high peaks (warmer sandstone in warm dry zones, slate in cold alpine)
        float strataBand = rockPattern(vec2(vWorldPos.x * 0.08 + vWorldPos.z * 0.08, vWorldPos.y * 0.14));
        float coldRockT = (uHasClimate == 1) ? (1.0 - smoothstep(0.0, 14.0, tempC)) : smoothstep(0.30, 0.78, hNorm);
        vec3 rockCol = mix(rockWarm, rockSlate, coldRockT);
        rockCol = mix(rockDark, rockCol * 1.16, strataBand);
        // Darken rock crevices using fine rock noise + cavity AO
        rockCol *= (0.72 + 0.42 * curvAO);
        // High-altitude rocky scree even on moderate slopes
        float screeMask = clamp(rockMask + smoothstep(0.58, 0.82, hNorm) * 0.65, 0.0, 1.0);
        albedo = mix(albedo, rockCol, screeMask);

        // Snow accumulation (follows real 0..2°C freezing isotherm when climate is available!)
        float snowMask = 0.0;
        if (uHasClimate == 1) {
          float freezeT = 1.6 + (macroNoise - 0.5) * 2.2 + precipNorm * 1.4;
          snowMask = (1.0 - smoothstep(freezeT - 2.6, freezeT, tempC)) * smoothstep(20.0, 65.0, elev);
          snowMask *= (1.0 - smoothstep(0.20, 0.40, baseSlope));
        } else {
          float snowLine = 0.62 + (macroNoise - 0.5) * 0.10 + (1.0 - moisture) * 0.06;
          snowMask = smoothstep(snowLine, snowLine + 0.13, hNorm) * (1.0 - smoothstep(0.18, 0.36, baseSlope));
        }
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
          float specAmt = (wetSpec * 0.65 + snowMask * 0.22 + rockMask * 0.08);
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
export function createWaterMaterial(heightFloatTex, maskTex, worldSpan) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uHeightTex: { value: heightFloatTex },
      uMasks: { value: maskTex },
      uWorldSpan: { value: worldSpan },
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
        pos.y += sin(pos.x * 0.045 + uTime * 1.7) * 0.24 + cos(pos.z * 0.055 - uTime * 1.35) * 0.20;
        vec4 wp = modelMatrix * vec4(pos, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uHeightTex;
      uniform sampler2D uMasks;
      uniform float uWorldSpan;
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
        vec2 uv = vWorldPos.xz / uWorldSpan + 0.5;
        uv.y = 1.0 - uv.y;
        bool inMap = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
        // Never draw the y=0 ocean sheet underneath an inland crater lake whose floor dips below sea level!
        if (inMap && texture2D(uMasks, uv).g > 0.05) discard;
        float terrainH = inMap
          ? texture2D(uHeightTex, uv).r * uVScale
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
        float foamRipple = sin(depthM * 3.8 - uTime * 3.2 + w1 * 3.0) * 0.5 + 0.5;
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
