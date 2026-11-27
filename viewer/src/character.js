/**
 * The explorer's body: a rigged glTF character driven by a shared set of animation clips.
 *
 * Assets live in /characters (packed by experiments/pack_characters.py from Quaternius's CC0
 * Universal Base Characters, Modular Character Outfits and Universal Animation Library). Bodies,
 * outfits, hairstyles and clips all use one skeleton with the same bone names, so:
 *   - a clip made on the library's mannequin plays on any body by name, and
 *   - a hairstyle (its own skinned mesh) is re-bound to the body's bones.
 *
 * A preset is an outfit + head + skin tone + hairstyles. Add one to CHARACTER_PRESETS to offer
 * another.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BASE = '/characters/';
const MODEL_HEIGHT_M = 1.8; // the kit's bodies, standing

export const CHARACTER_PRESETS = [
  {
    id: 'rowan',
    label: 'Rowan',
    outfit: 'Male_Ranger',
    head: 'Superhero_Male_FullBody',
    skin: 'T_Superhero_Male_Light',
    hair: ['Hair_SimpleParted', 'Hair_Beard'],
    hairColor: '#8a5a34',
    hide: /Head_Hood/, // hair and a raised hood do not fit together
  },
  { id: 'amara', label: 'Amara', outfit: 'Female_Ranger', outfitColor: 'T_Ranger_3_BaseColor', head: 'Superhero_Female_FullBody', hair: [], hairColor: '#3a2618' },
  { id: 'malik', label: 'Malik', outfit: 'Male_Peasant', head: 'Superhero_Male_FullBody', hair: ['Hair_Buzzed'], hairColor: '#2c2019' },
  {
    id: 'freya',
    label: 'Freya',
    outfit: 'Female_Peasant',
    outfitColor: 'T_Peasant_2_BaseColor',
    head: 'Superhero_Female_FullBody',
    skin: 'T_Superhero_Female_Light',
    hair: ['Hair_Long'],
    hairColor: '#e0b45e',
  },
];

/** Motion name -> clip in animations.glb */
const MOTIONS = {
  idle: 'Idle_Loop',
  walk: 'Walk_Loop',
  jog: 'Jog_Fwd_Loop',
  sprint: 'Sprint_Loop',
  swim: 'Swim_Fwd_Loop',
  tread: 'Swim_Idle_Loop',
  jump: 'Jump_Loop',
  sit: 'Sitting_Idle_Loop',
  drive: 'Driving_Loop',
};

const loader = new GLTFLoader();
let clipsPromise = null;

/**
 * Clips retargeted by bone name. Only rotations are kept, plus the pelvis position (the body's
 * bounce and lean): other positions and all scales would impose the mannequin's proportions.
 */
function loadClips() {
  if (!clipsPromise) {
    clipsPromise = loader.loadAsync(`${BASE}animations.glb`).then((gltf) => {
      const byName = new Map();
      for (const clip of gltf.animations) {
        clip.tracks = clip.tracks.filter((t) => t.name.endsWith('.quaternion') || t.name === 'pelvis.position');
        byName.set(clip.name, clip);
      }
      return byName;
    });
  }
  return clipsPromise;
}

const HEAD_BONES = new Set(['Head', 'neck_01']);

async function loadTexture(name) {
  const map = await new THREE.TextureLoader().loadAsync(`${BASE}tex/${name}.jpg`);
  map.flipY = false; // glTF convention
  map.colorSpace = THREE.SRGBColorSpace;
  return map;
}

/** Drop every triangle that is not on the head or neck (judged by each vertex's main bone). */
function keepHeadOnly(mesh) {
  const g = mesh.geometry;
  const { skinIndex, skinWeight } = g.attributes;
  const headIdx = new Set();
  mesh.skeleton.bones.forEach((b, i) => HEAD_BONES.has(b.name) && headIdx.add(i));
  const onHead = (v) => {
    let best = -1;
    let bone = 0;
    for (let k = 0; k < 4; k++) {
      const w = skinWeight.getComponent(v, k);
      if (w > best) {
        best = w;
        bone = skinIndex.getComponent(v, k);
      }
    }
    return headIdx.has(bone);
  };
  const src = g.index.array;
  const kept = [];
  for (let i = 0; i < src.length; i += 3) {
    if (onHead(src[i]) && onHead(src[i + 1]) && onHead(src[i + 2])) kept.push(src[i], src[i + 1], src[i + 2]);
  }
  g.setIndex(kept);
}

/** Re-bind a skinned mesh from its own skeleton to `bones` (same rig, matched by name). */
function adopt(mesh, bones, anchor) {
  const target = mesh.skeleton.bones.map((b) => bones.get(b.name));
  if (target.some((b) => !b)) return false; // not this rig
  anchor.add(mesh);
  mesh.bind(new THREE.Skeleton(target, mesh.skeleton.boneInverses), mesh.bindMatrix);
  mesh.castShadow = true;
  mesh.frustumCulled = false; // skinned bounds are the bind pose's; the animated body leaves them
  return true;
}

function skinnedMeshes(root) {
  const out = [];
  root.traverse((o) => o.isSkinnedMesh && out.push(o));
  return out;
}

/**
 * A preset is assembled on the outfit's skeleton: the outfit is a headless clothed body, and
 * the head (cut from a base body), eyes, eyebrows and hair are re-bound to its bones.
 */
async function loadBody(preset) {
  const [outfit, base, skinMap, clothMap, ...hair] = await Promise.all([
    loader.loadAsync(`${BASE}${preset.outfit}.gltf`),
    loader.loadAsync(`${BASE}${preset.head}.gltf`),
    preset.skin ? loadTexture(preset.skin) : null,
    preset.outfitColor ? loadTexture(preset.outfitColor) : null,
    ...(preset.hair || []).map((name) => loader.loadAsync(`${BASE}${name}.gltf`)),
  ]);
  const root = outfit.scene;
  const bones = new Map();
  root.traverse((o) => o.isBone && bones.set(o.name, o));
  let anchor = root;
  const setMap = (material, map) => {
    material.map = map;
    material.needsUpdate = true;
  };
  for (const mesh of skinnedMeshes(root)) {
    anchor = mesh.parent.isSkinnedMesh ? anchor : mesh.parent;
    mesh.castShadow = true;
    mesh.frustumCulled = false;
    if (preset.hide?.test(mesh.name) || preset.hide?.test(mesh.parent.name)) mesh.visible = false;
    const name = mesh.material.name || '';
    // Bare arms on the outfit share the base bodies' skin layout
    if (skinMap && /MI_Regular_/.test(name)) setMap(mesh.material, skinMap);
    if (clothMap && /MI_(Ranger|Peasant)/.test(name)) setMap(mesh.material, clothMap);
  }
  for (const mesh of skinnedMeshes(base.scene)) {
    const isBody = /MI_Superhero_/.test(mesh.material.name || '');
    if (isBody) {
      keepHeadOnly(mesh);
      if (skinMap) setMap(mesh.material, skinMap);
    }
    adopt(mesh, bones, anchor);
  }
  for (const h of hair) {
    for (const mesh of skinnedMeshes(h.scene)) adopt(mesh, bones, anchor);
  }
  // The kit's hair and eyebrow textures are greyscale, made to be tinted
  if (preset.hairColor) {
    for (const mesh of skinnedMeshes(root)) {
      if (/MI_Hair/.test(mesh.material.name || '')) mesh.material.color.set(preset.hairColor);
    }
  }
  return root;
}

function disposeModel(root) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry.dispose();
    for (const m of [].concat(o.material)) {
      for (const key of ['map', 'normalMap', 'roughnessMap', 'metalnessMap']) m[key]?.dispose();
      m.dispose();
    }
  });
}

/**
 * @param {object} opts
 * @param {THREE.Object3D} opts.parent   group that is positioned and turned as the player
 * @param {number} opts.height           standing height in world units
 * @param {() => void} [opts.onReady]    called each time a preset has finished loading
 */
export function createCharacter({ parent, height, onReady }) {
  const holder = new THREE.Group();
  holder.scale.setScalar(height / MODEL_HEIGHT_M);
  parent.add(holder);

  let model = null;
  let mixer = null;
  let actions = new Map();
  let current = null;
  let motion = 'idle';
  let presetIndex = -1;
  let loadToken = 0;

  function play(name, fade) {
    const next = actions.get(name) || actions.get('idle');
    if (!next || next === current) return;
    next.reset().setEffectiveWeight(1).fadeIn(fade).play();
    if (current) current.fadeOut(fade);
    current = next;
  }

  async function setPreset(index) {
    const i = ((index % CHARACTER_PRESETS.length) + CHARACTER_PRESETS.length) % CHARACTER_PRESETS.length;
    if (i === presetIndex) return;
    presetIndex = i;
    const token = ++loadToken;
    const [clips, root] = await Promise.all([loadClips(), loadBody(CHARACTER_PRESETS[i])]);
    if (token !== loadToken) {
      disposeModel(root);
      return;
    }
    if (model) {
      mixer.stopAllAction();
      holder.remove(model);
      disposeModel(model);
    }
    model = root;
    holder.add(model);
    mixer = new THREE.AnimationMixer(model);
    actions = new Map();
    for (const [name, clipName] of Object.entries(MOTIONS)) {
      const clip = clips.get(clipName);
      if (clip) actions.set(name, mixer.clipAction(clip));
    }
    current = null;
    play(motion, 0);
    onReady?.();
  }

  return {
    get ready() {
      return Boolean(model);
    },
    get presetIndex() {
      return presetIndex;
    },
    holder,
    setPreset,
    /** @param {keyof MOTIONS} name  @param {number} [rate] playback speed */
    setMotion(name, rate = 1) {
      motion = name;
      if (!mixer) return;
      play(name, 0.22);
      if (current) current.timeScale = rate;
    },
    update(dt) {
      if (mixer) mixer.update(dt);
    },
  };
}
