import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { Figure } from "@/lib/manualy-api";
import { FigurePreview } from "./FigureCard";

export function FigureZoom({
  figure,
  onClose,
}: {
  figure: Figure | null;
  onClose: () => void;
}) {
  return (
    <Dialog open={figure !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl rounded-3xl border-border bg-card">
        {figure && (
          <>
            <DialogHeader>
              <DialogTitle className="font-display text-xl">{figure.caption}</DialogTitle>
            </DialogHeader>
            <div className="flex items-center gap-2 text-xs">
              <span className="rounded-full bg-sky px-3 py-1 font-semibold text-sky-foreground">
                📄 Page {figure.page}
              </span>
              <span className="rounded-full bg-pink px-3 py-1 font-semibold text-pink-foreground">
                {figure.kind}
              </span>
            </div>
            <div className="overflow-hidden rounded-2xl border border-border bg-muted">
              <FigurePreview figure={figure} className="h-[26rem] w-full" />
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
