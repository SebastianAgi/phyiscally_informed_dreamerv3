import glob
import re
import os
from typing import List, Optional, Tuple, Iterable

import elements
import embodied
import numpy as np

try:
  from PIL import Image as _PILImage
except Exception:
  _PILImage = None


class PhysicsDataset(embodied.Env):
  """
  Image-only offline environment. Each episode is one subfolder under
  the provided `path` (e.g. `_run0`, `_run1`, ...). Images in each
  subfolder are sorted naturally and stepped through as an episode.

  Notes:
  - This class intentionally only supports image-mode datasets.
  - Actions are ignored except for a boolean 'reset'.
  - Image shape = (H, W, 3) uint8.
  """

  def __init__(
      self,
      task: str,
      path: Optional[str] = None,
      pattern: str = "*.png",
      size: Optional[tuple] = None,
      loop: bool = True,
      # Image-mode customization:
      subdir_prefix: Optional[str] = None,
      max_subdirs: Optional[int] = None,
      file_prefix: Optional[str] = None,
      exts: Optional[Tuple[str, ...]] = None,
  ):
    del task
    self._path = path
    self._pattern = pattern
    self._size = size  # (H, W) or None to infer from first frame
    self._loop = loop
    self._subdir_prefix = subdir_prefix
    self._max_subdirs = max_subdirs
    self._file_prefix = file_prefix

    # Normalize extensions to lower-case with leading dot; accept str or iterable
    if exts is None or exts == '':
      self._exts = None
    else:
      if isinstance(exts, str):
        parts = [p for p in re.split(r"[,\s]+", exts) if p]
      else:
        parts = list(exts)
      norm: List[str] = []
      for e in parts:
        e = str(e).strip().lower()
        if not e:
          continue
        if not e.startswith('.'):
          e = f'.{e}'
        norm.append(e)
      self._exts = tuple(norm) if norm else None

    # Discover image episodes (each subdir is an episode)
    self._episodes: List[List[str]] = []
    # Parallel lists for mass mask files per episode (may be empty lists)
    self._mask_episodes: List[List[str]] = []
    print('path', self._path)
    if not path or not os.path.isdir(path):
      raise FileNotFoundError(f"Image dataset path not found or not a directory: '{path}'")
    self._episodes, self._mask_episodes = self._discover_image_episodes(path, pattern)
    if not self._episodes:
      filt_desc = f"prefix='{self._file_prefix or '*'}' exts='{self._exts or pattern}'"
      raise FileNotFoundError(f"No image files found under '{path}' with {filt_desc}.")

    # Indices/state
    self._img_idx = -1  # index within current image episode
    self._ep_idx = -1   # current episode index (image mode)
    self._done = True
    self._first = True

    # Probe size from first image if not provided
    h, w = self._probe_image_size(self._episodes[0][0])
    if self._size is None:
      self._size = (h, w)

  # --- Spaces ---
  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'mass': elements.Space(np.uint8, self._size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        # Minimal continuous action so DictConcat has a non-empty space.
        'noop': elements.Space(np.float32, (1,), low=None, high=None),
    }

  # --- Stepping ---
  def step(self, action):
    # Only image-mode supported: each episode is one subfolder.
    if action.pop('reset') or self._done:
      self._first = True
      self._done = False
      self._open_next_episode()
      out = self._next_image()
      assert out is not None, 'Dataset had no frames.'
      frame, mask = out
      return self._obs(frame, mask, is_first=True)

    out = self._next_image()
    if out is None:
      self._done = True
      blank = np.zeros(self._size + (3,), np.uint8)
      blank_mask = np.zeros(self._size + (3,), np.uint8)
      return self._obs(blank, blank_mask, is_last=True, is_terminal=True)
    frame, mask = out
    return self._obs(frame, mask)

  # --- Helpers ---
  def _discover_files(self, path: Optional[str], pattern: str) -> List[str]:
    if path is None:
      return []
    if os.path.isdir(path):
      files = sorted(glob.glob(os.path.join(path, pattern)))
    elif os.path.isfile(path):
      files = [path]
    else:
      # Allow glob patterns directly
      files = sorted(glob.glob(path))
    return files


  def _discover_image_episodes(self, path: Optional[str], pattern: str) -> List[List[str]]:
    episodes: List[List[str]] = []
    mask_episodes: List[List[str]] = []
    if path is None:
      return episodes, mask_episodes
    if os.path.isdir(path):
      # Subdirectories become episodes; if none, the directory itself is an episode.
      raw = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
      if self._subdir_prefix:
        raw = [d for d in raw if d.startswith(self._subdir_prefix)]
      subdirs = [os.path.join(path, d) for d in self._natural_sorted(raw)]
      if self._max_subdirs is not None:
        subdirs = subdirs[: int(self._max_subdirs)]
      if subdirs:
        for d in subdirs:
          files = self._list_images(d, pattern)
          masks = self._list_masks(d)
          if files:
            episodes.append(files)
            mask_episodes.append(masks)
      else:
        files = self._list_images(path, pattern)
        masks = self._list_masks(path)
        if files:
          episodes.append(files)
          mask_episodes.append(masks)
    elif os.path.isfile(path):
      episodes.append([path])
      mask_episodes.append([])
    else:
      # Glob directly; group by parent folder as episodes.
      files = sorted(glob.glob(path, recursive=True))
      bydir = {}
      for f in files:
        d = os.path.dirname(f)
        bydir.setdefault(d, []).append(f)
      for d in self._natural_sorted(list(bydir.keys())):
        episodes.append(self._natural_sorted_fullpaths(bydir[d]))
        mask_episodes.append(self._list_masks(d))
    return episodes, mask_episodes

  def _list_images(self, folder: str, pattern: str) -> List[str]:
    # Start from pattern match to reduce IO, then refine by prefix/ext filters.
    candidates = glob.glob(os.path.join(folder, pattern))
    if not candidates:
      return []
    out = []
    for p in candidates:
      base = os.path.basename(p)
      name, ext = os.path.splitext(base)
      if self._file_prefix and not base.startswith(self._file_prefix):
        continue
      if self._exts is not None and ext.lower() not in self._exts:
        continue
      out.append(p)
    # If no additional filters provided, out==candidates; in any case, natural sort.
    return self._natural_sorted_fullpaths(out)

  def _list_masks(self, folder: str) -> List[str]:
    # Find mask files named like mass_mask_0000.png (any ext in self._exts)
    # Prefer natural sort. If no masks present, return empty list.
    if not os.path.isdir(folder):
      return []
    # Use pattern matching for the expected mask prefix
    candidates = sorted(glob.glob(os.path.join(folder, 'mass_*')))
    out = []
    for p in candidates:
      base = os.path.basename(p)
      name, ext = os.path.splitext(base)
      if self._file_prefix and not base.startswith(self._file_prefix):
        continue
      if self._exts is not None and ext.lower() not in self._exts:
        continue
      out.append(p)
    return self._natural_sorted_fullpaths(out)

  @staticmethod
  def _num_key(text: str) -> Tuple:
    # Extract all numbers for natural sorting; fallback to text for stable order.
    import re as _re
    nums = [int(x) for x in _re.findall(r"\d+", text)]
    return (*nums, text)

  def _natural_sorted(self, names: List[str]) -> List[str]:
    return sorted(names, key=self._num_key)

  def _natural_sorted_fullpaths(self, files: List[str]) -> List[str]:
    return sorted(files, key=lambda p: self._num_key(os.path.basename(p)))
  
  def _probe_image_size(self, filepath: str) -> Tuple[int, int]:
    if _PILImage is None:
      raise ImportError("Pillow is required for image datasets. Please install 'pillow'.")
    with _PILImage.open(filepath) as im:
      im = im.convert('RGB')
      return im.height, im.width

  def _next_image(self):
    # Iterate within the current image episode (subfolder)
    if self._ep_idx < 0:
      self._open_next_episode()
    files = self._episodes[self._ep_idx]
    self._img_idx += 1
    if self._img_idx >= len(files):
      # Episode finished
      return None
    path = files[self._img_idx]
    if _PILImage is None:
      raise ImportError("Pillow is required for image datasets. Please install 'pillow'.")
    with _PILImage.open(path) as im:
      im = im.convert('RGB')
      img = np.array(im)
    if (img.shape[0], img.shape[1]) != self._size:
      img = self._resize_nn(img, self._size)
    # Load corresponding mass mask if available for this episode
    mask = None
    if self._mask_episodes and len(self._mask_episodes) > self._ep_idx:
      mask_files = self._mask_episodes[self._ep_idx]
      if self._img_idx < len(mask_files):
        mpath = mask_files[self._img_idx]
        with _PILImage.open(mpath) as mim:
          mim = mim.convert('L')
          m = np.array(mim)
        if (m.shape[0], m.shape[1]) != self._size:
          m = self._resize_nn(m[..., None], self._size)[..., 0]
        # Normalize to float32 [0,1] and add channel dim
        mask = (m.astype(np.uint8) * 256 / 1000.0)[..., None]
        mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
    if mask is None:
      mask = np.zeros(self._size + (3,), dtype=np.uint8)
    return img, mask

  def _open_next_episode(self):
    # Advance to next subfolder episode in image mode.
    n = len(self._episodes)
    if n == 0:
      raise RuntimeError('No image episodes available.')
    self._ep_idx = (self._ep_idx + 1) % n
    self._img_idx = -1

  def _resize_nn(self, image: np.ndarray, size: tuple) -> np.ndarray:
    # Prefer Pillow's optimized nearest-neighbor resize when available
    # (Pillow implements the operation in C and is typically much faster
    # than a pure-NumPy Python implementation). Fall back to a simple
    # NumPy-based nearest-neighbor index sampling when Pillow isn't present.
    h, w = image.shape[:2]
    nh, nw = size
    if _PILImage is not None:
      # PIL expects size as (width, height)
      im = _PILImage.fromarray(image)
      im = im.resize((nw, nh), resample=_PILImage.NEAREST)
      return np.array(im)
    # Fallback: simple nearest-neighbor sampling using NumPy indexing.
    ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
    xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
    return image[ys][:, xs]

  def _obs(self, image, mass, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.asarray(image, np.uint8),
        mass=np.asarray(mass, np.uint8),
        reward=np.float32(0.0),
        is_first=bool(is_first),
        is_last=bool(is_last),
        is_terminal=bool(is_terminal),
    )
