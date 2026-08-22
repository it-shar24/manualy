import type { Figure } from "@/lib/manualy-api";
import { FigureCard } from "./FigureCard";

export function VisualIndex({
  figures,
  onZoom,
}: {
  figures: Figure[];
  onZoom: (figure: Figure) => void;
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-lg font-bold">✨ Visual index</h2>
        <span className="text-xs text-muted-foreground">
          {figures.length > 0 ? "Click a card to zoom" : "Upload a manual to populate"}
        </span>
      </div>

      {figures.length === 0 ? (
        <div className="card-float flex h-32 items-center justify-center px-6 text-center text-xs text-muted-foreground">
          Extracted schematics and diagrams will appear here.
        </div>
      ) : (
        <div className="-mx-1 flex snap-x gap-3 overflow-x-auto px-1 pb-3">
          {figures.map((figure) => (
            <FigureCard key={figure.id} figure={figure} onZoom={onZoom} className="snap-start" />
          ))}
        </div>
      )}
    </section>
  );
}
