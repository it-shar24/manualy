import { useEffect, useRef, useState } from "react";
import { Loader2, Send, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatResult, Figure } from "@/lib/manualy-api";
import { FigureCard } from "./FigureCard";

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | ({ id: string; role: "assistant" } & ChatResult);

const SUGGESTIONS = [
  "How do I replace the filter?",
  "Explain the wiring schematic",
  "What does error code E3 mean?",
];

function CitationTag({ page, label }: { page: number; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-butter px-2.5 py-1 text-[11px] font-bold text-butter-foreground">
      📄 Page {page} ({label})
    </span>
  );
}

export function ChatWorkspace({
  messages,
  pending,
  disabled,
  onSend,
  onZoom,
}: {
  messages: ChatMessage[];
  pending: boolean;
  disabled: boolean;
  onSend: (query: string) => void;
  onZoom: (figure: Figure) => void;
}) {
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const submit = () => {
    const query = input.trim();
    if (!query || pending) return;
    onSend(query);
    setInput("");
  };

  return (
    <section className="card-float flex min-h-0 flex-1 flex-col overflow-hidden">
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <h2 className="font-display text-lg font-bold">💬 Workspace</h2>
        <span className="rounded-full bg-secondary px-3 py-1 text-[11px] font-bold text-secondary-foreground">
          Grounded answers only
        </span>
      </header>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
        {messages.length === 0 && (
          <div className="mx-auto max-w-md space-y-4 pt-6 text-center">
            <p className="font-display text-xl">Ask anything about your manual</p>
            <p className="text-sm text-muted-foreground">
              Answers come back with page citations and the diagrams they came from.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => onSend(suggestion)}
                  disabled={disabled}
                  className="rounded-full border border-border bg-card px-3 py-1.5 text-xs font-semibold text-card-foreground transition-colors hover:bg-secondary disabled:opacity-50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="flex justify-end">
              <p className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-[var(--shadow-soft)]">
                {message.text}
              </p>
            </div>
          ) : (
            <div key={message.id} className="max-w-[92%] space-y-3">
              {message.outOfScope ? (
                <div className="flex gap-3 rounded-2xl border border-amber-soft-border bg-amber-soft p-4">
                  <TriangleAlert className="mt-0.5 size-5 shrink-0 text-amber-soft-foreground" />
                  <div className="space-y-1">
                    <p className="font-display text-sm font-bold text-amber-soft-foreground">
                      Out of scope
                    </p>
                    <p className="text-sm text-amber-soft-foreground/90">
                      {message.answer ||
                        "This question isn't covered by the uploaded manual, so there's nothing to cite."}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {message.answer}
                </p>
              )}

              {message.citations.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {message.citations.map((citation, index) => (
                    <CitationTag
                      key={`${citation.page}-${index}`}
                      page={citation.page}
                      label={citation.label}
                    />
                  ))}
                </div>
              )}

              {message.figures.length > 0 && (
                <div className="flex flex-wrap gap-3 rounded-2xl bg-muted/70 p-3">
                  {message.figures.map((figure) => (
                    <FigureCard key={figure.id} figure={figure} onZoom={onZoom} />
                  ))}
                </div>
              )}
            </div>
          ),
        )}

        {pending && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Reading the manual…
          </p>
        )}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="flex items-end gap-2 border-t border-border bg-card px-4 py-3"
      >
        <textarea
          value={input}
          rows={1}
          placeholder={disabled ? "Upload a manual to start chatting" : "Ask about a step, part or diagram…"}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          className="max-h-32 min-h-11 flex-1 resize-none rounded-2xl border border-input bg-background px-4 py-3 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
        />
        <button
          type="submit"
          disabled={pending || input.trim().length === 0}
          className={cn(
            "flex size-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-[var(--shadow-float)] transition-opacity",
            "disabled:opacity-50",
          )}
          aria-label="Send message"
        >
          <Send className="size-4" />
        </button>
      </form>
    </section>
  );
}
