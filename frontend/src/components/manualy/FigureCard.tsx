import { ImageOff } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Figure } from "@/lib/manualy-api";

export function FigurePreview({
  figure,
  className,
}: {
  figure: Figure;
  className?: string;
}) {
  if (!figure.imageUrl) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 bg-muted text-muted-foreground",
          className,
        )}
      >
        <ImageOff className="size-6" />
        <span className="text-xs">No preview available</span>
      </div>
    );
  }
  return (
    <img
      src={figure.imageUrl}
      alt={figure.caption}
      loading="lazy"
      className={cn("bg-card object-contain", className)}
    />
  );
}

export function FigureCard({
  figure,
  onZoom,
  className,
}: {
  figure: Figure;
  onZoom: (figure: Figure) => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onZoom(figure)}
      className={cn(
        "card-float group w-56 shrink-0 overflow-hidden p-2 text-left transition-transform hover:-translate-y-1",
        className,
      )}
    >
      <div className="overflow-hidden rounded-lg border border-border">
        <FigurePreview figure={figure} className="h-32 w-full" />
      </div>
      <div className="mt-2 space-y-1.5 px-1 pb-1">
        <p className="line-clamp-2 text-xs font-semibold leading-snug text-card-foreground">
          {figure.caption}
        </p>
        <div className="flex items-center gap-1.5">
          <span className="rounded-full bg-sky px-2 py-0.5 text-[10px] font-bold text-sky-foreground">
            Page {figure.page}
          </span>
          <span className="rounded-full bg-mint px-2 py-0.5 text-[10px] font-bold text-mint-foreground">
            {figure.kind}
          </span>
        </div>
      </div>
    </button>
  );
}
