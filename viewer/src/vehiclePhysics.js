/**
 * Vehicle movement rules, free of three.js and the DOM so they can be unit-tested under node.
 *
 * A vehicle's state is { kind, x, y, z, heading, speed, vy, pitch, roll, age }. `pitch` is
 * nose-up positive and `roll` is right-side-down positive, both in radians. `heading` is the
 * direction of travel as atan2(dx, dz): forward is (sin h, cos h), the same convention as
 * `player.yaw`. Steering right DEcreases it (x is east, z is south, seen from above).
 *
 * The world is reached through a `terrain` object:
 *   height(x, z)            ground height (viewer metres, sea level 0)
 *   lake(x, z)              lake mask 0..1
 *   move(x, z, nx, nz)      -> { x, z, blocked }: clamps a step to the loaded, walkable world
 */

export const VEHICLES = {
  boat: { label: 'Boat', key: 'b', medium: 'water', maxSpeed: 95, reverse: 25, accel: 38, coast: 0.5, turn: 1.0, steerNeedsSpeed: true, activity: 'SAILING', rider: 'sit' },
  horse: { label: 'Horse', key: 'r', medium: 'land', maxSpeed: 62, boost: 1.75, reverse: 14, accel: 85, coast: 2.2, turn: 2.3, maxSlope: 40, wadeDepth: 1.6, activity: 'RIDING', rider: 'sit' },
  jeep: { label: 'Jeep', key: 'j', medium: 'land', maxSpeed: 135, reverse: 38, accel: 52, coast: 0.9, turn: 1.35, steerNeedsSpeed: true, maxSlope: 31, wadeDepth: 0.7, activity: 'DRIVING', rider: 'drive' },
  glider: { label: 'Glider', key: 'g', medium: 'air', cruise: 72, dive: 128, flare: 42, sinkCruise: 6.5, sinkDive: 24, sinkFlare: 3.2, turn: 0.85, launch: 34, activity: 'GLIDING', rider: 'swim' },
};

const SEA_DEPTH = -0.25; // ground below this is under open water
const LAKE_MASK = 0.4;

export function isWater(terrain, x, z) {
  return terrain.height(x, z) < SEA_DEPTH || terrain.lake(x, z) > LAKE_MASK;
}

/** Height of the water surface at a point that `isWater`. */
export function waterLevel(terrain, x, z) {
  const h = terrain.height(x, z);
  return h < SEA_DEPTH ? 0 : h + 0.5;
}

export function slopeDegAlong(terrain, x, z, heading, reach = 6) {
  const dx = Math.sin(heading) * reach;
  const dz = Math.cos(heading) * reach;
  return (Math.atan2(terrain.height(x + dx, z + dz) - terrain.height(x - dx, z - dz), 2 * reach) * 180) / Math.PI;
}

/** Nearest point of the wanted kind within `radius`, searching outwards; null if none. */
export function findNearby(terrain, x, z, wantWater, radius = 90, step = 10) {
  if (isWater(terrain, x, z) === wantWater) return { x, z };
  for (let r = step; r <= radius; r += step) {
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      const px = x + Math.sin(a) * r;
      const pz = z + Math.cos(a) * r;
      if (isWater(terrain, px, pz) === wantWater) return { x: px, z: pz };
    }
  }
  return null;
}

/**
 * Where a vehicle summoned by a player at (x, z) appears.
 * @returns {{ok: true, x: number, z: number} | {ok: false, reason: string}}
 */
export function summonSpot(kind, terrain, x, z) {
  const cfg = VEHICLES[kind];
  if (!cfg) return { ok: false, reason: `Unknown vehicle ${kind}` };
  if (cfg.medium === 'water') {
    const spot = findNearby(terrain, x, z, true);
    return spot ? { ok: true, ...spot } : { ok: false, reason: 'A boat needs water: none within reach' };
  }
  const spot = findNearby(terrain, x, z, false, cfg.medium === 'air' ? 0 : 90);
  if (spot) return { ok: true, ...spot };
  return { ok: false, reason: cfg.medium === 'air' ? 'A glider launches from land' : `A ${cfg.label.toLowerCase()} needs dry land: none within reach` };
}

export function createVehicleState(kind, x, z, heading, terrain) {
  const cfg = VEHICLES[kind];
  const ground = terrain.height(x, z);
  const s = { kind, x, z, y: ground, heading, speed: 0, vy: 0, pitch: 0, roll: 0, age: 0, blocked: false, landed: false };
  if (cfg.medium === 'water') s.y = waterLevel(terrain, x, z);
  if (cfg.medium === 'air') {
    // A launch kick, so a glider is usable from flat ground as well as from a summit
    s.y = ground + 3;
    s.vy = cfg.launch;
    s.speed = cfg.cruise * 0.6;
  }
  return s;
}

function approach(value, target, maxDelta) {
  return value < target ? Math.min(target, value + maxDelta) : Math.max(target, value - maxDelta);
}

function stepSurface(s, cfg, input, dt, terrain) {
  const top = cfg.maxSpeed * (input.boost && cfg.boost ? cfg.boost : 1);
  if (input.throttle > 0) s.speed = approach(s.speed, top, cfg.accel * dt);
  else if (input.throttle < 0) s.speed = approach(s.speed, -cfg.reverse, cfg.accel * 1.6 * dt);
  else s.speed *= Math.exp(-cfg.coast * dt);
  if (Math.abs(s.speed) < 0.05 && input.throttle === 0) s.speed = 0;

  // Wheels and rudders only bite when moving; a horse can turn on the spot
  const grip = cfg.steerNeedsSpeed ? Math.min(1, Math.abs(s.speed) / (cfg.maxSpeed * 0.2)) * Math.sign(s.speed || 1) : 1;
  const turned = -input.steer * cfg.turn * grip * dt;
  s.heading += turned;

  s.blocked = false;
  if (s.speed !== 0) {
    const nx = s.x + Math.sin(s.heading) * s.speed * dt;
    const nz = s.z + Math.cos(s.heading) * s.speed * dt;
    let allowed;
    if (cfg.medium === 'water') {
      allowed = isWater(terrain, nx, nz);
    } else {
      const h = terrain.height(nx, nz);
      const tooDeep = h < -cfg.wadeDepth;
      const climb = slopeDegAlong(terrain, nx, nz, s.heading) * Math.sign(s.speed);
      allowed = !tooDeep && climb <= cfg.maxSlope;
    }
    if (allowed) {
      const step = terrain.move(s.x, s.z, nx, nz);
      s.x = step.x;
      s.z = step.z;
      if (step.blocked) {
        s.blocked = true;
        s.speed *= 0.5;
      }
    } else {
      s.blocked = true;
      s.speed = 0;
    }
  }

  if (cfg.medium === 'water') {
    s.y = waterLevel(terrain, s.x, s.z);
    s.pitch = 0;
    s.roll = approach(s.roll, input.steer * 0.12 * Math.min(1, Math.abs(s.speed) / cfg.maxSpeed), dt);
  } else {
    s.y = Math.max(terrain.height(s.x, s.z), -cfg.wadeDepth);
    const k = Math.min(1, dt * 8);
    s.pitch += ((slopeDegAlong(terrain, s.x, s.z, s.heading, 4) * Math.PI) / 180 - s.pitch) * k;
    // heading - 90° points to the vehicle's right: ground rising there lifts the right side
    s.roll += ((-slopeDegAlong(terrain, s.x, s.z, s.heading - Math.PI / 2, 2.5) * Math.PI) / 180 - s.roll) * k;
  }
  return turned;
}

function stepGlider(s, cfg, input, dt, terrain) {
  // Stick forward dives (fast, steep); back flares (slow, floaty)
  const targetSpeed = input.throttle > 0 ? cfg.dive : input.throttle < 0 ? cfg.flare : cfg.cruise;
  let sink = input.throttle > 0 ? cfg.sinkDive : input.throttle < 0 ? cfg.sinkFlare : cfg.sinkCruise;
  s.speed = approach(s.speed, targetSpeed, 34 * dt);

  // Ridge lift: air pushed up a slope the glider is flying towards
  const ahead = terrain.height(s.x + Math.sin(s.heading) * 70, s.z + Math.cos(s.heading) * 70);
  const rise = Math.max(0, Math.min(0.5, (ahead - terrain.height(s.x, s.z)) / 70));
  sink -= rise * 16;

  s.vy += (-sink - s.vy) * Math.min(1, dt * 1.4);
  const turned = -input.steer * cfg.turn * dt;
  s.heading += turned;
  s.roll = approach(s.roll, input.steer * 0.5, dt * 1.6);
  s.pitch = approach(s.pitch, Math.atan2(s.vy, s.speed) * 0.7, dt * 1.5);

  const step = terrain.move(s.x, s.z, s.x + Math.sin(s.heading) * s.speed * dt, s.z + Math.cos(s.heading) * s.speed * dt);
  s.x = step.x;
  s.z = step.z;
  s.blocked = step.blocked;
  s.y += s.vy * dt;

  const floor = Math.max(terrain.height(s.x, s.z), 0) + 1.2;
  if (s.y <= floor) {
    s.y = floor;
    if (s.age > 0.6) s.landed = true;
  }
  return turned;
}

/**
 * Advance a vehicle by dt.
 * @param {{throttle: number, steer: number, boost: boolean}} input  throttle and steer in -1..1
 * @returns {number} the change in heading, so a chase camera can turn with the vehicle
 */
export function stepVehicle(s, input, dt, terrain) {
  const cfg = VEHICLES[s.kind];
  s.age += dt;
  return cfg.medium === 'air' ? stepGlider(s, cfg, input, dt, terrain) : stepSurface(s, cfg, input, dt, terrain);
}
