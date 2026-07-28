import { useState } from 'react';
import { motion } from 'framer-motion';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';

type ImageItem = {
  img: string;
  title?: string;
  subtitle?: string;
  metadata?: Record<string, unknown>;
};

type Props = {
  items: ImageItem[];
  cols?: number;
};

export default function ImageGrid({ items, cols = 3 }: Props) {
  const [selected, setSelected] = useState<ImageItem | null>(null);

  if (!items || items.length === 0) {
    return <p className="text-sm text-muted-foreground py-4">No images available.</p>;
  }

  const gridClass =
    cols === 4
      ? 'grid grid-cols-2 gap-3 sm:grid-cols-4'
      : 'grid grid-cols-2 gap-3 sm:grid-cols-3';

  return (
    <>
      <div className={gridClass}>
        {items.map((item, idx) => (
          <motion.figure
            key={idx}
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.25, delay: Math.min(idx, 12) * 0.025, ease: 'easeOut' }}
            className="group cursor-pointer overflow-hidden rounded-lg border border-border bg-card shadow-sm transition hover:border-primary/50 hover:shadow-card"
            onClick={() => setSelected(item)}
          >
            <div className="aspect-square overflow-hidden bg-secondary">
              <img
                src={item.img}
                alt={item.title ?? 'image'}
                className="h-full w-full object-cover transition group-hover:scale-105"
                loading="lazy"
              />
            </div>
            {(item.title || item.subtitle) && (
              <figcaption className="px-2 py-1.5">
                {item.title && (
                  <p className="truncate text-xs font-medium text-foreground">{item.title}</p>
                )}
                {item.subtitle && (
                  <p className="truncate text-[11px] text-muted-foreground">{item.subtitle}</p>
                )}
              </figcaption>
            )}
          </motion.figure>
        ))}
      </div>

      <Dialog open={!!selected} onOpenChange={(open) => { if (!open) setSelected(null); }}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-base">{selected?.title ?? 'Image'}</DialogTitle>
          </DialogHeader>
          {selected && (
            <div className="space-y-4">
              <img
                src={selected.img}
                alt={selected.title ?? 'preview'}
                className="w-full rounded-lg border border-border object-contain max-h-[60vh]"
              />
              {selected.subtitle && (
                <p className="text-sm text-muted-foreground">{selected.subtitle}</p>
              )}
              {selected.metadata && Object.keys(selected.metadata).length > 0 && (
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {Object.entries(selected.metadata).map(([k, v]) => (
                    <div key={k} className="flex gap-2">
                      <dt className="text-muted-foreground capitalize">{k.replace(/_/g, ' ')}:</dt>
                      <dd className="font-medium text-foreground">{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
