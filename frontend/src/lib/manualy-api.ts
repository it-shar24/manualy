export const API_BASE_URL =
  (import.meta.env['VITE_API_BASE_URL'] as string | undefined) ?? "http://localhost:8000";

export type Figure = {
  id: string;
  page: number;
  caption: string;
  imageUrl: string | null;
  kind: string;
};

export type Citation = {
  page: number;
  label: string;
};

export type UploadResult = {
  fileName: string;
  totalPages: number;
  figures: Figure[];
};

export type ChatResult = {
  answer: string;
  citations: Citation[];
  figures: Figure[];
  outOfScope: boolean;
};

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};

const absolute = (url: unknown): string | null => {
  if (typeof url !== "string" || url.length === 0) return null;
  if (/^(https?:|data:)/.test(url)) return url;
  return `${API_BASE_URL}${url.startsWith("/") ? "" : "/"}${url}`;
};

const toFigure = (raw: unknown, index: number): Figure => {
  const r = asRecord(raw);
  return {
    id: String(r['id'] ?? r['figure_id'] ?? `figure-${index}`),
    page: Number(r['page'] ?? r['page_number'] ?? 0),
    caption: String(r['caption'] ?? r['title'] ?? r['description'] ?? "Extracted figure"),
    imageUrl: absolute(r['image_url'] ?? r['url'] ?? r['image'] ?? r['image_base64']),
    kind: String(r['type'] ?? r['kind'] ?? "Schematic"),
  };
};

const toFigures = (raw: unknown): Figure[] =>
  Array.isArray(raw) ? raw.map(toFigure) : [];

const toCitations = (raw: unknown): Citation[] => {
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    if (typeof item === "number") return { page: item, label: "Text" };
    const r = asRecord(item);
    return {
      page: Number(r['page'] ?? r['page_number'] ?? 0),
      label: String(r['label'] ?? r['type'] ?? r['source'] ?? "Text"),
    };
  });
};

export async function uploadManual(file: File): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, { method: "POST", body });
  if (!response.ok) {
    throw new Error(`Upload failed (${response.status})`);
  }
  const data = asRecord(await response.json());
  const figures = toFigures(data['figures'] ?? data['images'] ?? data['visual_index']);
  return {
    fileName: String(data['filename'] ?? data['file_name'] ?? file.name),
    totalPages: Number(data['total_pages'] ?? data['pages'] ?? 0),
    figures,
  };
}

export async function askManual(query: string): Promise<ChatResult> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    throw new Error(`Chat failed (${response.status})`);
  }
  const data = asRecord(await response.json());
  const answer = String(data['answer'] ?? data['response'] ?? data['message'] ?? "");
  const scopeFlag = data['out_of_scope'] ?? data['outOfScope'] ?? data['is_out_of_scope'];
  return {
    answer,
    citations: toCitations(data['citations'] ?? data['sources']),
    figures: toFigures(data['figures'] ?? data['images']),
    outOfScope: scopeFlag === true || String(data['scope'] ?? "") === "out_of_scope",
  };
}
