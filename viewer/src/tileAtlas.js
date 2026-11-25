import * as THREE from 'three';

/**
 * Phase-1 ghost rim: low-res neighbor planes around the home tile (not walkable).
 * Positions use home-local XZ: (qi * worldSpan, qj * worldSpan).
 */
export function createGhostTileGroup() {
  const group = new THREE.Group();
  group.name = 'ghostTiles';
  return group;
}

export function disposeGhostTileGroup(group) {
  if (!group) return;
  group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    }
  });
  while (group.children.length) group.remove(group.children[0]);
}

/**
 * Build a displaced plane from a Float32 heightfield (row-major, size×size).
 * Heights are viewer meters (same as home world).
 */
export function buildGhostTileMesh(heights, size, worldSpan, qi, qj, vScale = 1) {
  const segs = Math.min(size - 1, 128);
  const geo = new THREE.PlaneGeometry(worldSpan, worldSpan, segs, segs);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const tmp = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    tmp.fromBufferAttribute(pos, i);
    // PlaneGeometry spans [-half, half] in x/z after rotate
    const u = tmp.x / worldSpan + 0.5;
    const v = tmp.z / worldSpan + 0.5;
    const gx = Math.min(size - 1, Math.max(0, u * (size - 1)));
    const gz = Math.min(size - 1, Math.max(0, v * (size - 1)));
    const x0 = Math.floor(gx);
    const z0 = Math.floor(gz);
    const x1 = Math.min(size - 1, x0 + 1);
    const z1 = Math.min(size - 1, z0 + 1);
    const tx = gx - x0;
    const tz = gz - z0;
    const h00 = heights[z0 * size + x0];
    const h10 = heights[z0 * size + x1];
    const h01 = heights[z1 * size + x0];
    const h11 = heights[z1 * size + x1];
    const h = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
    pos.setY(i, Math.max(h, -40) * vScale);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  const mat = new THREE.MeshLambertMaterial({
    color: '#6b7c8a',
    flatShading: true,
    transparent: true,
    opacity: 0.72,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(qi * worldSpan, 0, qj * worldSpan);
  mesh.userData.tileQi = qi;
  mesh.userData.tileQj = qj;
  mesh.frustumCulled = false;
  return mesh;
}
