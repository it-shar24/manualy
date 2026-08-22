import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { UploadSidebar } from "@/components/manualy/UploadSidebar";
import { VisualIndex } from "@/components/manualy/VisualIndex";
import { ChatWorkspace, type ChatMessage } from "@/components/manualy/ChatWorkspace";
import { FigureZoom } from "@/components/manualy/FigureZoom";
import { askManual, uploadManual, type Figure, type UploadResult } from "@/lib/manualy-api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Manualy — Ask your PDF manuals anything" },
      {
        name: "description",
        content:
          "Manualy is a pastel document intelligence workspace: upload a PDF manual, browse extracted schematics, and get answers with page citations.",
      },
      { property: "og:title", content: "Manualy — Ask your PDF manuals anything" },
      {
        property: "og:description",
        content:
          "Upload a PDF manual, browse its visual index of diagrams, and chat with cited, grounded answers.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ManualyWorkspace,
});

const newId = () => Math.random().toString(36).slice(2);

function ManualyWorkspace() {
  const [document, setDocument] = useState<UploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [zoomed, setZoomed] = useState<Figure | null>(null);

  const handleUpload = async (file: File) => {
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      toast.error("Please upload a PDF manual");
      return;
    }
    setUploading(true);
    try {
      const result = await uploadManual(file);
      setDocument(result);
      setMessages([]);
      toast.success(`${result.fileName} indexed`, {
        description: `${result.totalPages} pages · ${result.figures.length} figures`,
      });
    } catch (error) {
      toast.error("Upload failed", {
        description: error instanceof Error ? error.message : "Could not reach the API",
      });
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async (query: string) => {
    setMessages((prev) => [...prev, { id: newId(), role: "user", text: query }]);
    setPending(true);
    try {
      const result = await askManual(query);
      setMessages((prev) => [...prev, { id: newId(), role: "assistant", ...result }]);
    } catch (error) {
      toast.error("Chat failed", {
        description: error instanceof Error ? error.message : "Could not reach the API",
      });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="gingham-bg min-h-screen">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <UploadSidebar document={document} uploading={uploading} onUpload={handleUpload} />

        <main className="flex min-h-screen flex-1 flex-col gap-5 p-5 lg:h-screen lg:min-h-0 lg:overflow-hidden">
          <VisualIndex figures={document?.figures ?? []} onZoom={setZoomed} />
          <ChatWorkspace
            messages={messages}
            pending={pending}
            disabled={uploading}
            onSend={handleSend}
            onZoom={setZoomed}
          />
        </main>
      </div>

      <FigureZoom figure={zoomed} onClose={() => setZoomed(null)} />
    </div>
  );
}
