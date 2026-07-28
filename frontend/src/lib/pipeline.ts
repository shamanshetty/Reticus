/**
 * Pipeline stage registry — single source of truth for the sphere navigator
 * and the page router. Thumbnails are generated locally as SVG data-URIs so
 * the navigator pulls zero external images (matches the "100% local" promise).
 */

export type PageKey = 'analysis' | 'preprocessing' | 'training' | 'results' | 'inference' | 'gradcam' | 'compare' | 'history';

type Stage = {
  key: PageKey;
  /** sphere card title */
  label: string;
  /** mono eyebrow on the card */
  tag: string;
  /** one-line description under the title */
  blurb: string;
  /** accent for the thumbnail glyph */
  color: string;
  /** single-glyph SVG path (24x24 viewBox) drawn into the thumb */
  glyph: string;
  /** generated data-URI */
  thumb: string;
};

// Lucide-derived glyph paths (24x24), one per stage - keeps the sphere on-brand
// with the sidebar icons that used to live here.
const GLYPHS: Record<PageKey, string> = {
  analysis: 'M3 3v18h18 M7 16l4-5 3 3 5-7',
  preprocessing: 'M4 7h10 M4 12h16 M4 17h10 M16 5l4 2-4 2 M6 15l-4 2 4 2',
  training: 'M3 12h4l2 6 4-12 2 6h6',
  results: 'M3 5h18 M3 12h18 M3 19h18 M9 5v14',
  inference: 'M12 3l2.4 5.6L20 11l-5.6 2.4L12 19l-2.4-5.6L4 11l5.6-2.4z',
  gradcam: 'M12 22c4-3 7-7 7-11a7 7 0 1 0-14 0c0 4 3 8 7 11z M12 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  compare: 'M8 3v18 M16 3v18 M3 8h5 M16 8h5 M3 16h5 M16 16h5',
  history: 'M12 8v4l3 3 M12 3a9 9 0 1 0 9 9',
};

const ACCENTS: Record<PageKey, string> = {
  analysis: '#40B5AD',
  preprocessing: '#9B59B6',
  training: '#50C878',
  results: '#FFB703',
  inference: '#40B5AD',
  gradcam: '#DC143C',
  compare: '#708090',
  history: '#5C6BC0',
};

function makeThumb(key: PageKey, dark: boolean): string {
  const color = ACCENTS[key];
  const glyph = GLYPHS[key];
  // Theme-aware disc face - SVG can't read CSS vars, so we bake the palette in
  // and regenerate when the theme flips. Glyph-only; title/desc live in the overlay.
  const inner = dark ? '#272c35' : '#ffffff';
  const outer = dark ? '#181b20' : '#f1efe7';
  const ring = dark ? 0.28 : 0.5;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="g" cx="50%" cy="42%" r="62%">
      <stop offset="0%" stop-color="${inner}"/>
      <stop offset="100%" stop-color="${outer}"/>
    </radialGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <circle cx="256" cy="256" r="252" fill="none" stroke="${color}" stroke-opacity="${ring}" stroke-width="3"/>
  <g transform="translate(256 256) scale(11) translate(-12 -12)" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="${glyph}"/>
  </g>
</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

/** Build the sphere disc image for a stage in the given theme. */
export function stageThumb(key: PageKey, dark: boolean): string {
  return makeThumb(key, dark);
}

const RAW: Array<Omit<Stage, 'thumb' | 'color' | 'glyph'>> = [
  { key: 'analysis', label: 'Data', tag: 'Stage 01', blurb: 'Class balance, duplicates, corrupt images.' },
  { key: 'preprocessing', label: 'Preprocessing', tag: 'Stage 02', blurb: 'Transform pipeline and before/after samples.' },
  { key: 'training', label: 'Training', tag: 'Stage 03', blurb: 'Loss / accuracy curves, ETA, system metrics.' },
  { key: 'results', label: 'Results', tag: 'Stage 04', blurb: 'Per-class metrics and PR curves.' },
  { key: 'inference', label: 'Inference', tag: 'Stage 05', blurb: 'Browse predictions with pagination.' },
  { key: 'gradcam', label: 'Grad-CAM', tag: 'Stage 06', blurb: 'Heatmaps over what the model attends to.' },
  { key: 'compare', label: 'Compare', tag: 'Stage 07', blurb: 'Overlay runs, diff configs, delta classes.' },
  { key: 'history', label: 'History', tag: 'Stage 08', blurb: 'Every run, when it ran, and what happened.' },
];

export const PIPELINE: Stage[] = RAW.map((s) => ({
  ...s,
  color: ACCENTS[s.key],
  glyph: GLYPHS[s.key],
  thumb: makeThumb(s.key, true),
}));
