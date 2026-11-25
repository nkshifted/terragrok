import * as THREE from 'three';

/**
 * Builds rich, multi-layered procedural world props driven by height, slope,
 * moisture, and river masks:
 *   1. Multi-tier Conifer / Pine Trees (3 stacked irregular foliage whorls + trunk)
 *   2. Broadleaf Valley Trees (organic multi-lobe canopy in moist warm lowlands)
 *   3. Craggy Cliff Boulders & Riverbank Rocks (displaced faceted rocks on steep
 *      slopes, ridges, and river channels to break up heightmap silhouettes)
 */
export function buildWorldProps(worldData, vScale = 1.0) {
  const group = new THREE.Group();
  const { size, worldSpan, maxElev, hasClimate, tempRange, precipRange } = worldData.meta;
  const climateSize = worldData.meta.climateSize || size;
  const { heights, masks, climate } = worldData;
  const cellSize = worldSpan / (size - 1);
  const spanFactor = Math.min(Math.max(worldSpan / 2400, 1.0), 2.1);
  const maxTrees = Math.round(5200 * spanFactor);
  const maxRocks = Math.round(3400 * spanFactor);

  const sampleClimateAt = (gx, gy) => {
    if (!hasClimate || !climate) return null;
    const cu = Math.max(0, Math.min(climateSize - 1, Math.round((gx / Math.max(size - 1, 1)) * (climateSize - 1))));
    const cv = Math.max(0, Math.min(climateSize - 1, Math.round((gy / Math.max(size - 1, 1)) * (climateSize - 1))));
    const cIdx = (cv * climateSize + cu) * 4;
    const tempC = tempRange[0] + (climate[cIdx] / 255) * (tempRange[1] - tempRange[0]);
    const pMm = precipRange[0] + (climate[cIdx + 1] / 255) * (precipRange[1] - precipRange[0]);
    return { tempC, pMm };
  };

  let s = 4242;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  const pines = [];
  const broadleafs = [];
  const rocks = [];

  const pl = worldData.meta.landmarks?.plains;
  const spx = pl ? (pl[0] - 0.5) * worldSpan : 99999;
  const spz = pl ? (pl[1] - 0.5) * worldSpan : 99999;

  // --- Scatter Trees (climate-driven timberline & species!) ---
  let attempts = 0;
  while (pines.length + broadleafs.length < maxTrees && attempts < maxTrees * 16) {
    attempts++;
    const gx = Math.floor(rand() * (size - 4)) + 2;
    const gy = Math.floor(rand() * (size - 4)) + 2;
    const idx = gy * size + gx;
    const elev = heights[idx];
    const mIdx = idx * 4;
    const river = masks[mIdx] / 255;
    const lake = masks[mIdx + 1] / 255;
    const moisture = masks[mIdx + 3] / 255;

    const hNorm = elev / Math.max(maxElev, 1);
    if (elev < 5.5 || lake > 0.2 || river > 0.26) continue;

    let tempC = 14.0 - hNorm * 18.0;
    let precipNorm = moisture;
    const clim = sampleClimateAt(gx, gy);
    if (clim) {
      tempC = clim.tempC;
      precipNorm = Math.min(Math.max(clim.pMm / 650, 0), 1);
      // Natural freezing timberline (~2.0 C) & true desert/arid cutoff (<260 mm/yr -> 0 trees!)
      if (tempC < 2.0 || clim.pMm < 260.0 || moisture < 0.16) continue;
      const treeProb = THREE.MathUtils.smoothstep(Math.max(precipNorm, moisture * 0.75), 0.24, 0.70) * 0.92;
      if (rand() > treeProb) continue;
    } else {
      if (hNorm > 0.56) continue;
      if (rand() > THREE.MathUtils.smoothstep(moisture, 0.25, 0.72) * 0.90) continue;
    }

    const dzx = (heights[idx + 1] - heights[idx - 1]) / (2 * cellSize);
    const dzy = (heights[idx + size] - heights[idx - size]) / (2 * cellSize);
    const slopeDeg = THREE.MathUtils.radToDeg(Math.atan(Math.hypot(dzx, dzy)));
    if (slopeDeg > 27) continue;

    const wx = (gx / (size - 1) - 0.5) * worldSpan + (rand() - 0.5) * cellSize;
    const wz = (gy / (size - 1) - 0.5) * worldSpan + (rand() - 0.5) * cellSize;
    if (Math.hypot(wx - spx, wz - spz) < 20) continue;

    const scale = (0.72 + rand() * 0.78) * (1.0 - hNorm * 0.38);
    const item = { wx, wz, baseElev: elev, scale, moisture, hNorm, rotY: rand() * Math.PI * 2 };

    // Warm moist zones get lush broadleafs; cool/boreal slopes get multi-tier conifers
    if (clim) {
      if (tempC > 8.5 && Math.max(precipNorm, moisture) > 0.36 && rand() < 0.64) {
        broadleafs.push(item);
      } else {
        pines.push(item);
      }
    } else if (hNorm < 0.26 && moisture > 0.48 && rand() < 0.55) {
      broadleafs.push(item);
    } else {
      pines.push(item);
    }
  }

  // --- Scatter Craggy Boulders (Cliff bases, mountain ridges & riverbanks!) ---
  attempts = 0;
  while (rocks.length < maxRocks && attempts < maxRocks * 22) {
    attempts++;
    const gx = Math.floor(rand() * (size - 4)) + 2;
    const gy = Math.floor(rand() * (size - 4)) + 2;
    const idx = gy * size + gx;
    const elev = heights[idx];
    const mIdx = idx * 4;
    const river = masks[mIdx] / 255;
    const lake = masks[mIdx + 1] / 255;
    if (elev < 1.5 || lake > 0.4) continue;

    const dzx = (heights[idx + 1] - heights[idx - 1]) / (2 * cellSize);
    const dzy = (heights[idx + size] - heights[idx - size]) / (2 * cellSize);
    const slopeDeg = THREE.MathUtils.radToDeg(Math.atan(Math.hypot(dzx, dzy)));
    const hNorm = elev / Math.max(maxElev, 1);

    const nearRiver = river > 0.14 && river < 0.55;
    const tempRockC = masks[mIdx + 3] !== undefined ? (masks[mIdx + 3] / 255) * 60 - 20 : 12;
    const isGlacialSheet = tempRockC < -3.0 && slopeDeg < 34;
    if (isGlacialSheet) continue;
    const rockyZone = (slopeDeg > 23 && slopeDeg < 48) || hNorm > 0.48 || nearRiver;
    if (!rockyZone) continue;

    const wx = (gx / (size - 1) - 0.5) * worldSpan + (rand() - 0.5) * cellSize;
    const wz = (gy / (size - 1) - 0.5) * worldSpan + (rand() - 0.5) * cellSize;
    if (Math.hypot(wx - spx, wz - spz) < 12) continue;

    const baseScale = nearRiver ? 0.6 + rand() * 1.4 : 1.1 + rand() * 3.0;
    rocks.push({
      wx,
      wz,
      baseElev: elev,
      sx: baseScale * (0.8 + rand() * 0.6),
      sy: baseScale * (0.55 + rand() * 0.55),
      sz: baseScale * (0.8 + rand() * 0.6),
      rx: rand() * Math.PI,
      ry: rand() * Math.PI,
      rz: rand() * Math.PI,
      shade: nearRiver ? 0.52 + rand() * 0.18 : 0.68 + rand() * 0.28,
    });
  }

  // ---------- 1. Multi-Tier Conifers ----------
  const pineTrunkGeo = new THREE.CylinderGeometry(0.18, 0.40, 3.0, 6);
  pineTrunkGeo.translate(0, 1.5, 0);
  const pineTier1 = new THREE.ConeGeometry(2.2, 3.4, 7);
  pineTier1.translate(0, 3.3, 0);
  const pineTier2 = new THREE.ConeGeometry(1.7, 3.1, 7);
  pineTier2.translate(0, 5.1, 0);
  const pineTier3 = new THREE.ConeGeometry(1.1, 2.6, 7);
  pineTier3.translate(0, 6.7, 0);

  const barkMat = new THREE.MeshStandardMaterial({ color: '#563c28', roughness: 0.95 });
  const pineMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.85, flatShading: true });

  const pTrunks = new THREE.InstancedMesh(pineTrunkGeo, barkMat, pines.length);
  const pF1 = new THREE.InstancedMesh(pineTier1, pineMat, pines.length);
  const pF2 = new THREE.InstancedMesh(pineTier2, pineMat, pines.length);
  const pF3 = new THREE.InstancedMesh(pineTier3, pineMat, pines.length);
  [pTrunks, pF1, pF2, pF3].forEach((m) => {
    m.castShadow = true;
    m.receiveShadow = true;
  });

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const cPineLow = new THREE.Color('#2d6235');
  const cPineHigh = new THREE.Color('#204438');

  for (let i = 0; i < pines.length; i++) {
    const p = pines[i];
    dummy.position.set(p.wx, p.baseElev * vScale - 0.35, p.wz);
    dummy.rotation.set(0, p.rotY, 0);
    dummy.scale.setScalar(p.scale);
    dummy.updateMatrix();
    pTrunks.setMatrixAt(i, dummy.matrix);
    pF1.setMatrixAt(i, dummy.matrix);
    pF2.setMatrixAt(i, dummy.matrix);
    pF3.setMatrixAt(i, dummy.matrix);

    color.copy(cPineLow).lerp(cPineHigh, THREE.MathUtils.clamp(p.hNorm * 1.6, 0, 1));
    color.offsetHSL((rand() - 0.5) * 0.03, 0, (rand() - 0.5) * 0.07);
    pF1.setColorAt(i, color);
    pF2.setColorAt(i, color.clone().multiplyScalar(1.08));
    pF3.setColorAt(i, color.clone().multiplyScalar(1.16));
  }
  [pTrunks, pF1, pF2, pF3].forEach((m) => {
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  });

  // ---------- 2. Broadleaf Valley Trees ----------
  const bTrunkGeo = new THREE.CylinderGeometry(0.22, 0.45, 3.2, 6);
  bTrunkGeo.translate(0, 1.6, 0);
  const bCrownGeo = new THREE.IcosahedronGeometry(2.4, 1);
  // Slight organic squashing of canopy
  const bp = bCrownGeo.attributes.position;
  for (let i = 0; i < bp.count; i++) {
    const x = bp.getX(i), y = bp.getY(i), z = bp.getZ(i);
    const j = 1.0 + 0.14 * Math.sin(x * 2.1 + z * 1.7);
    bp.setXYZ(i, x * j, y * 0.78 * j + 4.4, z * j);
  }
  bCrownGeo.computeVertexNormals();

  const broadMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.88, flatShading: true });
  const bTrunks = new THREE.InstancedMesh(bTrunkGeo, barkMat, broadleafs.length);
  const bCrowns = new THREE.InstancedMesh(bCrownGeo, broadMat, broadleafs.length);
  bTrunks.castShadow = true;
  bCrowns.castShadow = true;
  bCrowns.receiveShadow = true;

  const cBroadWarm = new THREE.Color('#5a8c36');
  const cBroadLush = new THREE.Color('#357232');
  for (let i = 0; i < broadleafs.length; i++) {
    const p = broadleafs[i];
    dummy.position.set(p.wx, p.baseElev * vScale - 0.3, p.wz);
    dummy.rotation.set(0, p.rotY, 0);
    dummy.scale.setScalar(p.scale * 1.05);
    dummy.updateMatrix();
    bTrunks.setMatrixAt(i, dummy.matrix);
    bCrowns.setMatrixAt(i, dummy.matrix);

    color.copy(cBroadWarm).lerp(cBroadLush, p.moisture);
    color.offsetHSL((rand() - 0.5) * 0.04, 0, (rand() - 0.5) * 0.08);
    bCrowns.setColorAt(i, color);
  }
  bTrunks.instanceMatrix.needsUpdate = true;
  bCrowns.instanceMatrix.needsUpdate = true;
  if (bCrowns.instanceColor) bCrowns.instanceColor.needsUpdate = true;

  // ---------- 3. Craggy Cliff Boulders & River Rocks ----------
  const rockGeo = new THREE.DodecahedronGeometry(1.0, 1);
  const rp = rockGeo.attributes.position;
  for (let i = 0; i < rp.count; i++) {
    const x = rp.getX(i), y = rp.getY(i), z = rp.getZ(i);
    const n = 0.82 + 0.34 * Math.abs(Math.sin(x * 3.1 + y * 2.3) * Math.cos(z * 2.7 - x * 1.9));
    rp.setXYZ(i, x * n, y * n * 0.85, z * n);
  }
  rockGeo.computeVertexNormals();

  const rockMat = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    roughness: 0.92,
    metalness: 0.04,
    flatShading: true,
  });
  const rockMesh = new THREE.InstancedMesh(rockGeo, rockMat, rocks.length);
  rockMesh.castShadow = true;
  rockMesh.receiveShadow = true;

  for (let i = 0; i < rocks.length; i++) {
    const r = rocks[i];
    dummy.position.set(r.wx, r.baseElev * vScale - r.sy * 0.25, r.wz);
    dummy.rotation.set(r.rx, r.ry, r.rz);
    dummy.scale.set(r.sx, r.sy, r.sz);
    dummy.updateMatrix();
    rockMesh.setMatrixAt(i, dummy.matrix);

    color.setRGB(0.56 * r.shade, 0.54 * r.shade, 0.50 * r.shade);
    rockMesh.setColorAt(i, color);
  }
  rockMesh.instanceMatrix.needsUpdate = true;
  if (rockMesh.instanceColor) rockMesh.instanceColor.needsUpdate = true;

  group.add(pTrunks, pF1, pF2, pF3, bTrunks, bCrowns, rockMesh);
  group.userData = {
    pines,
    broadleafs,
    rocks,
    pineMeshes: [pTrunks, pF1, pF2, pF3],
    broadMeshes: [bTrunks, bCrowns],
    rockMesh,
  };
  return group;
}

export function updatePropsTransform(group, vScale, isDiorama) {
  if (!group || !group.userData) return;
  const { pines, broadleafs, rocks, pineMeshes, broadMeshes, rockMesh } = group.userData;
  const dummy = new THREE.Object3D();
  const scaleMul = isDiorama ? 1.3 : 1.0;

  for (let i = 0; i < pines.length; i++) {
    const p = pines[i];
    dummy.position.set(p.wx, p.baseElev * vScale - 0.35, p.wz);
    dummy.rotation.set(0, p.rotY, 0);
    dummy.scale.setScalar(p.scale * scaleMul);
    dummy.updateMatrix();
    for (const m of pineMeshes) m.setMatrixAt(i, dummy.matrix);
  }
  for (const m of pineMeshes) m.instanceMatrix.needsUpdate = true;

  for (let i = 0; i < broadleafs.length; i++) {
    const p = broadleafs[i];
    dummy.position.set(p.wx, p.baseElev * vScale - 0.3, p.wz);
    dummy.rotation.set(0, p.rotY, 0);
    dummy.scale.setScalar(p.scale * 1.05 * scaleMul);
    dummy.updateMatrix();
    for (const m of broadMeshes) m.setMatrixAt(i, dummy.matrix);
  }
  for (const m of broadMeshes) m.instanceMatrix.needsUpdate = true;

  for (let i = 0; i < rocks.length; i++) {
    const r = rocks[i];
    dummy.position.set(r.wx, r.baseElev * vScale - r.sy * 0.25, r.wz);
    dummy.rotation.set(r.rx, r.ry, r.rz);
    dummy.scale.set(r.sx, r.sy, r.sz);
    dummy.updateMatrix();
    rockMesh.setMatrixAt(i, dummy.matrix);
  }
  rockMesh.instanceMatrix.needsUpdate = true;
}
