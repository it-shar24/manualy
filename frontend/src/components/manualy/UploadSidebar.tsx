import { useRef, useState } from "react";
import { FileText, Images, Loader2, UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";
import type { UploadResult } from "@/lib/manualy-api";
import { API_BASE_URL } from "@/lib/manualy-api";

function MetricCard({
  icon,
  label,
  value,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone: "pink" | "sky";
}) {
  return (
    <div className="card-float flex items-center gap-3 p-3">
      <span
        className={cn(
          "flex size-9 items-center justify-center rounded-xl",
          tone === "pink" ? "bg-pink text-pink-foreground" : "bg-sky text-sky-foreground",
        )}
      >
        {icon}
      </span>
      <div>
        <p className="font-display text-xl leading-none text-card-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

export function UploadSidebar({
  document,
  uploading,
  onUpload,
}: {
  document: UploadResult | null;
  uploading: boolean;
  onUpload: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <aside className="flex w-full flex-col gap-5 border-border bg-sidebar/80 p-5 backdrop-blur-sm lg:h-screen lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:border-r">
      <header className="text-center">
        <h1 className="font-display text-5xl font-extrabold tracking-tight text-primary drop-shadow-sm">
          Manualy
        </h1>
        <p className="mt-1 text-xs font-medium text-muted-foreground">
          Your manuals, finally understandable.
        </p>
      </header>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) onUpload(file);
        }}
        className={cn(
          "rounded-2xl border-2 border-dashed p-6 text-center transition-colors",
          dragging ? "border-primary bg-secondary" : "border-border bg-card",
        )}
      >
        <span className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-secondary text-secondary-foreground">
          {uploading ? (
            <Loader2 className="size-6 animate-spin" />
          ) : (
            <UploadCloud className="size-6" />
          )}
        </span>
        <p className="mt-3 text-sm font-semibold">Drop your PDF manual</p>
        <p className="mt-1 text-xs text-muted-foreground">
          {uploading ? "Extracting pages and figures…" : "or pick a file from your device"}
        </p>
        <button
          type="button"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
          className="mt-4 inline-flex items-center justify-center rounded-full bg-primary px-4 py-2 text-xs font-bold text-primary-foreground shadow-[var(--shadow-float)] transition-opacity hover:opacity-90 disabled:opacity-60"
        >
          Choose PDF
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onUpload(file);
            event.target.value = "";
          }}
        />
      </div>

      <section className="space-y-3">
        <h2 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
          Document overview
        </h2>
        <p className="truncate rounded-xl bg-muted px-3 py-2 text-xs text-muted-foreground">
          {document ? document.fileName : "No manual uploaded yet"}
        </p>
        <MetricCard
          tone="sky"
          icon={<FileText className="size-4" />}
          label="Total pages"
          value={document ? String(document.totalPages) : "—"}
        />
        <MetricCard
          tone="pink"
          icon={<Images className="size-4" />}
          label="Extracted figures"
          value={document ? String(document.figures.length) : "—"}
        />
      </section>

      <p className="mt-auto text-[10px] leading-relaxed text-muted-foreground">
        API base: <span className="font-mono">{API_BASE_URL}</span>
      </p>
    </aside>
  );
}
