import test from 'node:test';
import assert from 'node:assert/strict';

import { VEHICLES, createVehicleState, isWater, stepVehicle, summonSpot } from '../src/vehiclePhysics.js';

/** Terrain from a height function; nothing blocks movement unless `wallX` is set. */
function world(height, { lake = () => 0, wallX = Infinity } = {}) {
  return {
    height,
    lake,
    move: (x, z, nx, nz) => (nx > wallX ? { x, z, blocked: true } : { x: nx, z: nz, blocked: false }),
  };
}

const FLAT = world(() => 10);
// Dry land for x < 0, sea deepening to the east
const COAST = world((x) => (x < 0 ? 5 : -Math.min(30, x * 0.5) - 0.3));
const DT = 1 / 60;

function run(s, terrain, input, seconds) {
  for (let i = 0; i < seconds / DT; i++) stepVehicle(s, { throttle: 0, steer: 0, boost: false, ...input }, DT, terrain);
  return s;
}

const EAST = Math.PI / 2; // heading is atan2(dx, dz): +x

test('every vehicle has a distinct summon key', () => {
  const keys = Object.values(VEHICLES).map((v) => v.key);
  assert.equal(new Set(keys).size, keys.length);
});

test('a jeep accelerates to its top speed in a straight line', () => {
  const s = run(createVehicleState('jeep', 0, 0, EAST, FLAT), FLAT, { throttle: 1 }, 6);
  assert.ok(Math.abs(s.speed - VEHICLES.jeep.maxSpeed) < 0.5);
  assert.ok(s.x > 400 && Math.abs(s.z) < 1e-6);
});

test('steering right turns clockwise seen from above (heading decreases)', () => {
  const s = run(createVehicleState('horse', 0, 0, 0, FLAT), FLAT, { throttle: 1, steer: 1 }, 0.5);
  assert.ok(s.heading < 0);
  // Heading 0 is +z (south); a right turn from there bends towards -x (west)
  assert.ok(s.x < 0);
});

test('a jeep cannot turn on the spot but a horse can', () => {
  const jeep = run(createVehicleState('jeep', 0, 0, 0, FLAT), FLAT, { steer: 1 }, 1);
  const horse = run(createVehicleState('horse', 0, 0, 0, FLAT), FLAT, { steer: 1 }, 1);
  assert.equal(jeep.heading, 0);
  assert.ok(horse.heading < -1);
});

test('a slope stops a jeep that a horse still climbs', () => {
  const hill = world((x) => Math.max(0, x) * Math.tan((36 * Math.PI) / 180) + 5);
  const jeep = run(createVehicleState('jeep', -50, 0, EAST, hill), hill, { throttle: 1 }, 6);
  const horse = run(createVehicleState('horse', -50, 0, EAST, hill), hill, { throttle: 1 }, 6);
  assert.ok(jeep.x < 12 && jeep.blocked, `jeep reached x=${jeep.x}`);
  assert.ok(horse.x > 100, `horse reached x=${horse.x}`);
  assert.ok(horse.pitch > 0.4, 'the horse is nose-up on the climb');
});

test('land vehicles stop at deep water and a boat stops at the shore', () => {
  const jeep = run(createVehicleState('jeep', -60, 0, EAST, COAST), COAST, { throttle: 1 }, 6);
  assert.ok(jeep.x < 3 && jeep.blocked);
  const boat = run(createVehicleState('boat', 80, 0, -EAST, COAST), COAST, { throttle: 1 }, 8);
  assert.ok(boat.x >= 0 && boat.x < 3 && boat.blocked, `boat reached x=${boat.x}`);
  assert.equal(boat.y, 0);
  assert.ok(isWater(COAST, boat.x, boat.z));
});

test('summoning respects the medium', () => {
  assert.equal(summonSpot('boat', FLAT, 0, 0).ok, false);
  const afloat = summonSpot('boat', COAST, -40, 0);
  assert.ok(afloat.ok && afloat.x >= 0, 'the boat appears on the nearest water');
  const ashore = summonSpot('jeep', COAST, 40, 0);
  assert.ok(ashore.ok && ashore.x < 0, 'the jeep appears on the nearest land');
  assert.equal(summonSpot('horse', COAST, 900, 0).ok, false);
  assert.equal(summonSpot('glider', COAST, 40, 0).ok, false);
  assert.equal(summonSpot('glider', FLAT, 0, 0).ok, true);
});

test('a lake floats a boat above its bed', () => {
  const lake = world(() => 40, { lake: (x) => (Math.abs(x) < 50 ? 1 : 0) });
  const s = createVehicleState('boat', 0, 0, EAST, lake);
  assert.ok(s.y > 40);
  run(s, lake, { throttle: 1 }, 6);
  assert.ok(s.x < 50, 'and keeps it inside the lake');
});

test('a glider climbs off the launch, glides down and lands', () => {
  const s = createVehicleState('glider', 0, 0, EAST, FLAT);
  let peak = s.y;
  let t = 0;
  while (!s.landed && t < 120) {
    stepVehicle(s, { throttle: 0, steer: 0, boost: false }, DT, FLAT);
    peak = Math.max(peak, s.y);
    t += DT;
  }
  assert.ok(peak > 10 + 15, `launch gained ${peak - 10} m`);
  assert.ok(s.landed, 'it comes down by itself');
  assert.ok(s.x > 200, `it covered ${s.x} m`);
});

test('diving a glider is faster and steeper than flaring', () => {
  const dive = run(createVehicleState('glider', 0, 0, EAST, world(() => -5000)), world(() => -5000), { throttle: 1 }, 8);
  const flare = run(createVehicleState('glider', 0, 0, EAST, world(() => -5000)), world(() => -5000), { throttle: -1 }, 8);
  assert.ok(dive.speed > flare.speed * 2);
  assert.ok(dive.vy < flare.vy - 10);
  assert.ok(dive.pitch < flare.pitch, 'nose lower in the dive');
});

test('the world edge stops a vehicle without teleporting it', () => {
  const edge = world(() => 10, { wallX: 30 });
  const s = run(createVehicleState('horse', 0, 0, EAST, edge), edge, { throttle: 1 }, 4);
  assert.ok(s.x <= 30 && s.x > 25 && s.blocked);
});
