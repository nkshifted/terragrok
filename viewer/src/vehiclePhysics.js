/**
 * Vehicle movement rules, free of three.js and the DOM so they can be unit-tested under node.
 *
 * A vehicle's state is { kind, x, y, z, heading, speed, vy, pitch, roll, air, blocked, ... }.
 * `pitch` is nose-up positive and `roll` is right-side-down positive, both in radians. `heading`
 * is the direction of travel as atan2(dx, dz): forward is (sin h, cos h), the same convention as
 * `player.yaw`. Steering right DEcreases it (x is east, z is south, seen from above).
 * `blocked` is false, 'terrain' (too steep, too deep, no water) or 'edge' (the loaded world ends).
 * `swamped` is set when a land vehicle comes down in water too deep for it: it is lost.
 *
 * The world is reached through a `terrain` object:
 *   height(x, z)            ground height (viewer metres, sea level 0)
 *   lake(x, z)              lake mask 0..1
 *   river(x, z)             river mask 0..1 (optional; only small craft use it)
 *   move(x, z, nx, nz)      -> { x, z, blocked }: clamps a step to the loaded, walkable world
 */

export const VEHICLES = {
  boat: { label: 'Boat', key: 'b', medium: 'water', maxSpeed: 95, reverse: 25, accel: 38, coast: 0.5, turn: 1.0, steerNeedsSpeed: true, activity: 'SAILING', rider: 'sit' },
  // Light enough for rivers as well as lakes and the sea, and turns on the spot
  canoe: { label: 'Canoe', key: 'k', medium: 'water', rivers: true, portage: 14, maxSpeed: 46, reverse: 20, accel: 30, coast: 0.9, turn: 1.7, activity: 'PADDLING', rider: 'sit' },
  horse: { label: 'Horse', key: 'r', medium: 'land', maxSpeed: 62, boost: 1.75, reverse: 14, accel: 85, coast: 2.2, turn: 2.3, maxSlope: 62, wadeDepth: 1.6, jump: 27, activity: 'RIDING', rider: 'sit' },
  jeep: { label: 'Jeep', key: 'j', medium: 'land', maxSpeed: 135, reverse: 38, accel: 52, coast: 0.9, turn: 1.35, steerNeedsSpeed: true, maxSlope: 50, wadeDepth: 0.7, jump: 21, activity: 'DRIVING', rider: 'drive' },
  glider: { label: 'Glider', key: 'g', medium: 'air', cruise: 72, dive: 128, flare: 42, sinkCruise: 6.5, sinkDive: 24, sinkFlare: 3.2, climb: 13, ceiling: 700, turn: 0.85, launch: 34, activity: 'GLIDING', rider: 'swim' },
};

const SEA_DEPTH = -0.25; // ground below this is under open water
const LAKE_MASK = 0.4;
// Rivers are a blurred mask: this low a value is the reedy margin of a creek, which a canoe
// still floats over (walking counts as "in the river" only above 0.42)
const RIVER_MASK = 0.12;
const GRAVITY = 58; // the same as the player's jump
// Ground falling away faster than this (m/s) under a moving vehicle leaves it in the air
const TAKEOFF_RATE = 25;

/** Open water or a lake; with `rivers`, a river channel too. */
export function isWater(terrain, x, z, rivers = false) {
  if (terrain.height(x, z) < SEA_DEPTH || terrain.lake(x, z) > LAKE_MASK) return true;
  return rivers && Boolean(terrain.river) && terrain.river(x, z) > RIVER_MASK;
}

/** Height of the water surface at a point that `isWater`. */
export function waterLevel(terrain, x, z) {
  const h = terrain.height(x, z);
  if (h < SEA_DEPTH) return 0;
  return h + (terrain.lake(x, z) > LAKE_MASK ? 0.5 : 0.3);
}

export function slopeDegAlong(terrain, x, z, heading, reach = 6) {
  const dx = Math.sin(heading) * reach;
  const dz = Math.cos(heading) * reach;
  return (Math.atan2(terrain.height(x + dx, z + dz) - terrain.height(x - dx, z - dz), 2 * reach) * 180) / Math.PI;
}

/** Nearest point of the wanted kind within `radius`, searching outwards; null if none. */
export function findNearby(terrain, x, z, wantWater, radius = 90, step = 10, rivers = false) {
  if (isWater(terrain, x, z, rivers) === wantWater) return { x, z };
  for (let r = step; r <= radius; r += step) {
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      const px = x + Math.sin(a) * r;
      const pz = z + Math.cos(a) * r;
      if (isWater(terrain, px, pz, rivers) === wantWater) return { x: px, z: pz };
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
    // Rivers are narrow: search for them in finer steps
    const spot = findNearby(terrain, x, z, true, 90, cfg.rivers ? 5 : 10, Boolean(cfg.rivers));
    if (spot) return { ok: true, ...spot };
    return { ok: false, reason: `A ${cfg.label.toLowerCase()} needs ${cfg.rivers ? 'a river, a lake or the sea' : 'a lake or the sea'}: none within reach` };
  }
  const spot = findNearby(terrain, x, z, false, cfg.medium === 'air' ? 0 : 90);
  if (spot) return { ok: true, ...spot };
  return { ok: false, reason: cfg.medium === 'air' ? 'A glider launches from land' : `A ${cfg.label.toLowerCase()} needs dry land: none within reach` };
}

export function createVehicleState(kind, x, z, heading, terrain) {
  const cfg = VEHICLES[kind];
  const ground = terrain.height(x, z);
  const s = { kind, x, z, y: ground, heading, speed: 0, vy: 0, vyGround: 0, bent: 0, pitch: 0, roll: 0, age: 0, air: false, lifting: false, blocked: false, landed: false, swamped: false };
  if (cfg.medium === 'water') s.y = waterLevel(terrain, x, z);
  if (cfg.medium === 'air') {
    // A launch kick, so a glider is usable from flat ground as well as from a summit
    s.y = ground + 3;
    s.vy = cfg.launch;
    s.speed = cfg.cruise * 0.6;
    s.air = true;
  }
  return s;
}

function approach(value, target, maxDelta) {
  return value < target ? Math.min(target, value + maxDelta) : Math.max(target, value - maxDelta);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/** Take a horizontal step through terrain.move; records an 'edge' block. */
function advance(s, terrain, nx, nz) {
  const step = terrain.move(s.x, s.z, nx, nz);
  s.x = step.x;
  s.z = step.z;
  if (step.blocked) {
    s.blocked = 'edge';
    s.speed *= 0.5;
  }
}

/** Is there water within `reach` further along `heading` from (x, z)? */
function waterAhead(terrain, x, z, heading, reach, rivers) {
  for (let d = 2; d <= reach; d += 2) {
    if (isWater(terrain, x + Math.sin(heading) * d, z + Math.cos(heading) * d, rivers)) return true;
  }
  return false;
}

function stepWater(s, cfg, input, dt, terrain) {
  if (s.speed !== 0) {
    const dist = s.speed * dt;
    const nx = s.x + Math.sin(s.heading) * dist;
    const nz = s.z + Math.cos(s.heading) * dist;
    if (isWater(terrain, nx, nz, cfg.rivers)) advance(s, terrain, nx, nz);
    else if (cfg.portage) {
      // A small craft in a narrow channel: follow the bend if the water turns, and scrape
      // over a gravel bar or a spit of bank when there is water again just beyond it
      const bend = [0.3, -0.3, 0.6, -0.6, 0.95, -0.95].find((a) =>
        isWater(terrain, s.x + Math.sin(s.heading + a) * dist, s.z + Math.cos(s.heading + a) * dist, cfg.rivers)
      );
      if (bend !== undefined && !input.steerKey) {
        const turn = bend * Math.min(1, dt * 6);
        s.heading += turn;
        s.bent += turn;
        advance(s, terrain, s.x + Math.sin(s.heading) * dist, s.z + Math.cos(s.heading) * dist);
      } else if (waterAhead(terrain, nx, nz, s.heading, cfg.portage, cfg.rivers)) {
        advance(s, terrain, nx, nz);
        s.speed *= Math.exp(-2.5 * dt); // dragging over the shallows
      } else {
        s.blocked = 'terrain';
        s.speed = 0;
      }
    } else {
      s.blocked = 'terrain';
      s.speed = 0;
    }
  }
  s.y = isWater(terrain, s.x, s.z, cfg.rivers) ? waterLevel(terrain, s.x, s.z) : terrain.height(s.x, s.z) + 0.15;
  s.pitch = 0;
  s.roll = approach(s.roll, input.steer * 0.12 * Math.min(1, Math.abs(s.speed) / cfg.maxSpeed), dt);
}

/** In the air after a jump, a crest or a drop: a ballistic arc with a little steering. */
function stepAirborne(s, cfg, dt, terrain) {
  s.vy -= GRAVITY * dt;
  if (s.speed !== 0) {
    const nx = s.x + Math.sin(s.heading) * s.speed * dt;
    const nz = s.z + Math.cos(s.heading) * s.speed * dt;
    // A rock face ahead: stop and drop. Open water is no obstacle to something in the air.
    if (terrain.height(nx, nz) > s.y + 2.5) {
      s.blocked = 'terrain';
      s.speed = 0;
    } else advance(s, terrain, nx, nz);
  }
  s.y += s.vy * dt;
  const bed = terrain.height(s.x, s.z);
  const ground = Math.max(bed, -cfg.wadeDepth);
  if (s.y <= ground) {
    s.y = ground;
    s.vy = 0;
    s.vyGround = 0;
    s.air = false;
    s.speed *= 0.92;
    if (bed < -cfg.wadeDepth) {
      // Came down in water it cannot drive or wade out of
      s.swamped = true;
      s.speed = 0;
    }
  }
  s.pitch = approach(s.pitch, Math.atan2(s.vy, Math.abs(s.speed) + 20) * 0.6, dt * 3);
  s.roll = approach(s.roll, 0, dt * 2);
}

function stepLand(s, cfg, input, dt, terrain) {
  const y0 = s.y;
  if (s.speed !== 0) {
    const nx = s.x + Math.sin(s.heading) * s.speed * dt;
    const nz = s.z + Math.cos(s.heading) * s.speed * dt;
    // Water it would wade into is a barrier; water at the foot of a drop is not (it goes over)
    const bedAhead = terrain.height(nx, nz);
    const tooDeep = bedAhead < -cfg.wadeDepth && s.y - bedAhead < 3 + cfg.wadeDepth;
    const climb = slopeDegAlong(terrain, nx, nz, s.heading) * Math.sign(s.speed);
    if (tooDeep || climb > cfg.maxSlope) {
      s.blocked = 'terrain';
      s.speed = 0;
    } else advance(s, terrain, nx, nz);
  }
  const ground = Math.max(terrain.height(s.x, s.z), -cfg.wadeDepth);
  // Where momentum alone would carry it this frame, against where the ground now is
  const coasting = y0 + s.vyGround * dt;
  if (input.action && cfg.jump) {
    s.air = true;
    s.vy = cfg.jump + clamp(s.vyGround, 0, Math.abs(s.speed) * 0.5);
    s.y = Math.max(ground, y0);
  } else if ((coasting - ground) / dt > TAKEOFF_RATE) {
    // Over a crest at speed, or off an edge: the ground has dropped away
    s.air = true;
    s.vy = clamp(s.vyGround, -30, Math.abs(s.speed) * 0.6);
    s.y = coasting;
  } else {
    s.y = ground;
    s.vyGround += ((ground - y0) / dt - s.vyGround) * Math.min(1, dt * 10);
  }
  const k = Math.min(1, dt * 8);
  s.pitch += ((slopeDegAlong(terrain, s.x, s.z, s.heading, 4) * Math.PI) / 180 - s.pitch) * k;
  // heading - 90° points to the vehicle's right: ground rising there lifts the right side
  s.roll += ((-slopeDegAlong(terrain, s.x, s.z, s.heading - Math.PI / 2, 2.5) * Math.PI) / 180 - s.roll) * k;
}

function stepSurface(s, cfg, input, dt, terrain) {
  const airborne = s.air;
  if (!airborne) {
    let top = cfg.maxSpeed * (input.boost && cfg.boost ? cfg.boost : 1);
    // Climbing costs speed: down to under half on the steepest slope it can manage
    if (cfg.maxSlope) top *= 1 - 0.55 * clamp((slopeDegAlong(terrain, s.x, s.z, s.heading) * Math.sign(s.speed || 1)) / cfg.maxSlope, 0, 1);
    if (input.throttle > 0) s.speed = approach(s.speed, top, cfg.accel * dt);
    else if (input.throttle < 0) s.speed = approach(s.speed, -cfg.reverse, cfg.accel * 1.6 * dt);
    else s.speed *= Math.exp(-cfg.coast * dt);
    if (Math.abs(s.speed) < 0.05 && input.throttle === 0) s.speed = 0;
  }

  // Wheels and rudders only bite when moving; a horse or a paddler can turn on the spot
  const grip = cfg.steerNeedsSpeed ? Math.min(1, Math.abs(s.speed) / (cfg.maxSpeed * 0.2)) * Math.sign(s.speed || 1) : 1;
  const turned = -input.steer * cfg.turn * grip * (airborne ? 0.3 : 1) * dt;
  s.heading += turned;

  s.blocked = false;
  if (cfg.medium === 'water') stepWater(s, cfg, input, dt, terrain);
  else if (airborne) stepAirborne(s, cfg, dt, terrain);
  else stepLand(s, cfg, input, dt, terrain);
  return turned;
}

function stepGlider(s, cfg, input, dt, terrain) {
  const ground = Math.max(terrain.height(s.x, s.z), 0);
  // Catching the wind: a steady climb at a little less airspeed, up to a ceiling above ground
  s.lifting = Boolean(input.action) && s.y - ground < cfg.ceiling;
  // Stick forward dives (fast, steep); back flares (slow, floaty)
  let targetSpeed = input.throttle > 0 ? cfg.dive : input.throttle < 0 ? cfg.flare : cfg.cruise;
  let sink = input.throttle > 0 ? cfg.sinkDive : input.throttle < 0 ? cfg.sinkFlare : cfg.sinkCruise;
  if (s.lifting) {
    sink = -cfg.climb;
    targetSpeed *= 0.85;
  }
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

  s.blocked = false;
  advance(s, terrain, s.x + Math.sin(s.heading) * s.speed * dt, s.z + Math.cos(s.heading) * s.speed * dt);
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
 * @param {{throttle: number, steer: number, steerKey?: boolean, boost: boolean, action: boolean}} input
 *        throttle and steer in -1..1; `steerKey` when the steering comes from a key (a canoe then
 *        stops following bends by itself); `action` is the jump key (a glider's "catch the wind")
 * @returns {number} the change in heading, so a chase camera can turn with the vehicle
 */
export function stepVehicle(s, input, dt, terrain) {
  const cfg = VEHICLES[s.kind];
  // A frame can report no elapsed time; rates divide by dt, and one NaN would stick for good
  if (!(dt > 0)) return 0;
  s.age += dt;
  s.bent = 0; // heading change this step that the vehicle made by itself (following a river)
  return cfg.medium === 'air' ? stepGlider(s, cfg, input, dt, terrain) : stepSurface(s, cfg, input, dt, terrain);
}
