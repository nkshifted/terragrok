/**
 * Summonable vehicles: their meshes and the mount / dismount bookkeeping. How each one moves
 * is in vehiclePhysics.js.
 *
 * One vehicle exists at a time. Summoning puts the player on it; dismounting leaves it parked
 * (a glider is packed away) until it is mounted again or another is summoned.
 *
 * The meshes here are built from primitives, front towards -Z like the explorer group, in
 * explorer scale (the character stands 3.3 units tall). Each builder returns
 * { group, seat, animate(state, dt, t) }; `seat` is where the rider's origin goes, in the
 * vehicle's frame. To use a downloaded model, replace a builder.
 */
import * as THREE from 'three';
import { VEHICLES, createVehicleState, findNearby, stepVehicle, summonSpot } from './vehiclePhysics.js';

const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.82, flatShading: true, ...extra });

function box(parent, w, h, d, material, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x, y, z);
  m.castShadow = true;
  parent.add(m);
  return m;
}

function tube(parent, radius, from, to, material) {
  const a = new THREE.Vector3(...from);
  const b = new THREE.Vector3(...to);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 6), material);
  m.position.copy(a).add(b).multiplyScalar(0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  m.castShadow = true;
  parent.add(m);
  return m;
}

/** A flat outline in X/Z (z negative = bow) extruded downwards from y = top. */
function hullSlab(outline, depth, top, material) {
  const shape = new THREE.Shape(outline.map(([x, z]) => new THREE.Vector2(x, z)));
  const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
  geo.rotateX(Math.PI / 2); // shape y -> z, extrusion -> downwards
  const m = new THREE.Mesh(geo, material);
  m.position.y = top;
  m.castShadow = true;
  return m;
}

function buildBoat() {
  const group = new THREE.Group();
  const outline = (s) => [
    [-1.7 * s, 4.3 * s],
    [1.7 * s, 4.3 * s],
    [1.9 * s, 1.0 * s],
    [1.3 * s, -3.0 * s],
    [0, -5.2 * s],
    [-1.3 * s, -3.0 * s],
    [-1.9 * s, 1.0 * s],
  ];
  group.add(hullSlab(outline(1), 1.5, 1.0, mat('#b9552f')));
  group.add(hullSlab(outline(1.06), 0.16, 1.08, mat('#f1e6cf')));
  group.add(hullSlab(outline(0.84), 0.1, 1.1, mat('#5b3a22'))); // the well of the boat
  const wood = mat('#c9a26b');
  box(group, 3.0, 0.18, 0.8, wood, 0, 1.25, 1.2);
  box(group, 2.6, 0.18, 0.7, wood, 0, 1.25, -1.6);
  box(group, 0.7, 1.3, 0.8, mat('#30363d'), 0, 1.5, 4.5); // outboard motor
  box(group, 0.16, 0.16, 1.5, mat('#30363d'), 0.5, 1.9, 3.6);
  return {
    group,
    seat: new THREE.Vector3(0, 0.35, 1.2),
    animate(state, dt, t) {
      const run = Math.min(1, Math.abs(state.speed) / VEHICLES.boat.maxSpeed);
      group.position.y += Math.sin(t * 1.7) * 0.12;
      group.rotation.x += Math.sin(t * 1.3) * 0.025 + run * 0.07; // the bow lifts under power
      group.rotation.z += Math.sin(t * 1.1 + 1) * 0.03;
    },
  };
}

function buildJeep() {
  const group = new THREE.Group();
  const paint = mat('#5f7a3c');
  const dark = mat('#23272b');
  box(group, 3.5, 0.9, 7.2, paint, 0, 1.75, 0);
  box(group, 3.3, 0.75, 2.3, paint, 0, 2.5, -2.4); // bonnet
  box(group, 3.5, 0.95, 0.22, paint, 0, 2.65, 3.5); // tailgate
  box(group, 0.22, 0.95, 3.4, paint, -1.64, 2.65, 1.8);
  box(group, 0.22, 0.95, 3.4, paint, 1.64, 2.65, 1.8);
  box(group, 3.6, 0.3, 0.5, dark, 0, 1.4, -3.75); // bumpers
  box(group, 3.6, 0.3, 0.5, dark, 0, 1.4, 3.75);
  const glass = mat('#9fd0e6', { transparent: true, opacity: 0.45 });
  const screen = box(group, 3.2, 1.2, 0.1, glass, 0, 3.45, -1.2);
  screen.rotation.x = -0.2;
  tube(group, 0.09, [-1.6, 2.9, -1.1], [-1.6, 4.1, -1.35], dark);
  tube(group, 0.09, [1.6, 2.9, -1.1], [1.6, 4.1, -1.35], dark);
  tube(group, 0.1, [-1.6, 3.0, 1.2], [-1.6, 4.6, 1.2], dark); // roll bar
  tube(group, 0.1, [1.6, 3.0, 1.2], [1.6, 4.6, 1.2], dark);
  tube(group, 0.1, [-1.6, 4.6, 1.2], [1.6, 4.6, 1.2], dark);
  const seatMat = mat('#3b2f26');
  for (const x of [-0.8, 0.8]) {
    box(group, 1.2, 0.3, 1.3, seatMat, x, 2.25, 0.5);
    box(group, 1.2, 1.5, 0.28, seatMat, x, 2.9, 1.2);
  }
  const wheelGeo = new THREE.CylinderGeometry(0.98, 0.98, 0.75, 12);
  wheelGeo.rotateZ(Math.PI / 2);
  const wheels = [];
  for (const [x, z] of [
    [-1.85, -2.45],
    [1.85, -2.45],
    [-1.85, 2.45],
    [1.85, 2.45],
  ]) {
    const w = new THREE.Mesh(wheelGeo, dark);
    w.position.set(x, 0.98, z);
    w.castShadow = true;
    group.add(w);
    wheels.push(w);
    box(w, 0.8, 0.5, 0.5, mat('#8d949b'), 0, 0, 0); // hub, so the spin is visible
  }
  const spare = new THREE.Mesh(wheelGeo, dark);
  spare.rotation.y = Math.PI / 2;
  spare.position.set(0, 2.7, 3.95);
  group.add(spare);
  const wheel = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.07, 6, 14), dark);
  wheel.position.set(-0.8, 3.05, -0.75);
  wheel.rotation.x = -0.5;
  group.add(wheel);
  for (const x of [-1.15, 1.15]) box(group, 0.5, 0.4, 0.1, mat('#fff3c4', { emissive: '#665b2a' }), x, 2.45, -3.6);
  return {
    group,
    seat: new THREE.Vector3(-0.8, 1.42, 0.55),
    animate(state, dt) {
      for (const w of wheels) w.rotation.x -= (state.speed / 0.98) * dt;
      wheel.rotation.z = -state.steer * 0.9;
    },
  };
}

function buildHorse() {
  const group = new THREE.Group();
  const coat = mat('#7a4a2a');
  const dark = mat('#2a1b12');
  const body = new THREE.Group();
  group.add(body);
  box(body, 1.35, 1.55, 3.7, coat, 0, 2.75, 0);
  box(body, 1.25, 1.35, 1.0, coat, 0, 2.85, -2.1); // chest
  const neck = box(body, 0.75, 1.9, 0.95, coat, 0, 3.75, -2.35);
  neck.rotation.x = 0.55;
  const head = box(body, 0.62, 0.7, 1.5, coat, 0, 4.45, -3.2);
  head.rotation.x = 0.35;
  box(body, 0.14, 0.3, 0.14, coat, -0.22, 4.95, -2.75);
  box(body, 0.14, 0.3, 0.14, coat, 0.22, 4.95, -2.75);
  const mane = box(body, 0.2, 1.7, 0.5, dark, 0, 4.05, -1.95);
  mane.rotation.x = 0.55;
  const tail = box(body, 0.26, 1.7, 0.26, dark, 0, 2.7, 2.05);
  tail.rotation.x = 0.35;
  box(body, 1.45, 0.22, 1.4, mat('#3a2a20'), 0, 3.6, 0.1); // saddle
  box(body, 1.5, 0.5, 0.2, mat('#3a2a20'), 0, 3.4, -0.45);
  const legs = [];
  for (const [x, z, phase] of [
    [-0.45, -1.5, 0],
    [0.45, -1.5, Math.PI],
    [-0.45, 1.5, Math.PI],
    [0.45, 1.5, 0],
  ]) {
    const hip = new THREE.Group();
    hip.position.set(x, 2.05, z);
    box(hip, 0.4, 2.05, 0.42, coat, 0, -1.0, 0);
    box(hip, 0.44, 0.3, 0.46, dark, 0, -1.9, 0);
    group.add(hip);
    legs.push({ hip, phase });
  }
  let gait = 0;
  return {
    group,
    seat: new THREE.Vector3(0, 2.72, 0.15),
    animate(state, dt) {
      const run = Math.min(1.6, Math.abs(state.speed) / VEHICLES.horse.maxSpeed);
      gait += dt * (4 + run * 9) * Math.sign(state.speed || 1);
      const swing = Math.min(0.75, run * 0.8);
      for (const { hip, phase } of legs) hip.rotation.x = Math.sin(gait + phase) * swing;
      body.position.y = Math.abs(Math.sin(gait)) * 0.16 * Math.min(1, run);
      tail.rotation.x = 0.35 + run * 0.5;
    },
    riderBob: () => body.position.y,
  };
}

function buildGlider() {
  const group = new THREE.Group();
  const Y = 2.7; // the wing rides above the pilot
  const pts = { nose: [0, Y, -3.4], tail: [0, Y, 2.6], l: [-9.5, Y + 0.7, 3.0], r: [9.5, Y + 0.7, 3.0], li: [-3.6, Y + 0.25, 2.2], ri: [3.6, Y + 0.25, 2.2] };
  const panel = (verts, color) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts.flat(), 3));
    geo.computeVertexNormals();
    const m = new THREE.Mesh(geo, mat(color, { side: THREE.DoubleSide, roughness: 0.6 }));
    m.castShadow = true;
    group.add(m);
  };
  panel([pts.nose, pts.li, pts.tail, pts.nose, pts.tail, pts.ri], '#f2c230');
  panel([pts.nose, pts.l, pts.li, pts.nose, pts.ri, pts.r], '#d9482f');
  const frame = mat('#c9ced3', { roughness: 0.4 });
  tube(group, 0.08, pts.nose, pts.tail, frame);
  tube(group, 0.08, pts.nose, pts.l, frame);
  tube(group, 0.08, pts.nose, pts.r, frame);
  tube(group, 0.07, [-5.2, Y + 0.4, 0.2], [5.2, Y + 0.4, 0.2], frame);
  tube(group, 0.06, [0, Y, -0.3], [-1.3, 0.2, -1.5], frame); // control frame
  tube(group, 0.06, [0, Y, -0.3], [1.3, 0.2, -1.5], frame);
  tube(group, 0.07, [-1.3, 0.2, -1.5], [1.3, 0.2, -1.5], frame);
  tube(group, 0.03, [0, Y, 0.3], [0, 0.5, 0.6], mat('#30363d')); // harness strap
  return { group, seat: new THREE.Vector3(0, 0, 0.4), animate() {} };
}

const BUILDERS = { boat: buildBoat, horse: buildHorse, jeep: buildJeep, glider: buildGlider };
const MOUNT_RANGE = 14;

/**
 * @param {object} opts
 * @param {THREE.Scene} opts.scene
 * @param {{height: Function, lake: Function, move: Function}} opts.terrain  see vehiclePhysics.js
 */
export function createVehicles({ scene, terrain }) {
  let model = null;
  let state = null;
  let riding = false;
  const seatWorld = new THREE.Vector3();

  function pose(t = 0, dt = 0) {
    const g = model.group;
    g.position.set(state.x, state.y, state.z);
    g.rotation.set(state.pitch, state.heading + Math.PI, -state.roll, 'YXZ');
    model.animate(state, dt, t);
    g.updateMatrixWorld(true);
  }

  function release() {
    if (model) {
      scene.remove(model.group);
      model.group.traverse((o) => {
        if (!o.isMesh) return;
        o.geometry.dispose();
        o.material.dispose();
      });
    }
    model = null;
    state = null;
    riding = false;
  }

  return {
    get riding() {
      return riding;
    },
    get kind() {
      return state?.kind || null;
    },
    get state() {
      return state;
    },
    release,

    /** Bring a vehicle to the player and put them on it. */
    summon(kind, x, z, heading) {
      const spot = summonSpot(kind, terrain, x, z);
      if (!spot.ok) return spot;
      release();
      state = createVehicleState(kind, spot.x, spot.z, heading, terrain);
      state.steer = 0;
      model = BUILDERS[kind]();
      scene.add(model.group);
      riding = true;
      pose();
      return { ok: true };
    },

    /** Mount the parked vehicle if the player is next to it. */
    mountNearby(x, z) {
      if (!state || riding) return false;
      if (Math.hypot(state.x - x, state.z - z) > MOUNT_RANGE) return false;
      riding = true;
      return true;
    },

    /**
     * Step off. Returns where the player ends up: beside a land vehicle, on the nearest shore
     * (or in the water) from a boat, or in mid-air from a glider.
     */
    dismount() {
      if (!riding) return null;
      const cfg = VEHICLES[state.kind];
      const out = { x: state.x, y: state.y, z: state.z, vy: 0, airborne: false, heading: state.heading };
      if (cfg.medium === 'air') {
        out.vy = state.vy;
        out.airborne = !state.landed;
        release();
        return out;
      }
      riding = false;
      state.speed = 0;
      if (cfg.medium === 'water') {
        const shore = findNearby(terrain, state.x, state.z, false, 40, 5);
        if (shore) Object.assign(out, shore);
      } else {
        // Step down on the vehicle's left
        const lx = state.x + Math.sin(state.heading + Math.PI / 2) * 4.5;
        const lz = state.z + Math.cos(state.heading + Math.PI / 2) * 4.5;
        const step = terrain.move(state.x, state.z, lx, lz);
        out.x = step.x;
        out.z = step.z;
      }
      out.y = Math.max(terrain.height(out.x, out.z), -0.4);
      return out;
    },

    /**
     * @param {{throttle: number, steer: number, boost: boolean}} input
     * @returns {{turned: number, landed: boolean} | null} null when nobody is riding
     */
    update(dt, input, t) {
      if (!state) return null;
      if (!riding) {
        pose(t, dt); // a parked boat keeps bobbing
        return null;
      }
      state.steer = input.steer;
      const turned = stepVehicle(state, input, dt, terrain);
      pose(t, dt);
      return { turned, landed: state.landed };
    },

    /** Put the rider group on the seat, turned and tilted with the vehicle. */
    placeRider(rider) {
      seatWorld.copy(model.seat);
      if (model.riderBob) seatWorld.y += model.riderBob();
      model.group.localToWorld(seatWorld);
      rider.position.copy(seatWorld);
      rider.rotation.copy(model.group.rotation);
    },
  };
}
