import type {
  HealthResponse,
  IngestResponse,
  QueryRequest,
  QueryResponse,
  Source,
} from "./types"

// Configurable at build time: VITE_API_BASE_URL=http://localhost:8000 locally,
// or your hosted Render URL in production. Falls back to localhost so
// `npm run dev` works out of the box against a local `uvicorn api:app`.
export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ||
  "http://localhost:8000"

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(message: string, status: number, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError(
      "Could not reach the Evidence AI backend. Is the API server running?",
      0,
    )
  }
  if (!res.ok) {
    let detail: unknown
    try {
      detail = await res.json()
    } catch {
      detail = await res.text().catch(() => undefined)
    }
    throw new ApiError(`Request to ${path} failed (${res.status})`, res.status, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: (backend?: string) =>
    request<HealthResponse>(`/health${backend ? `?backend=${backend}` : ""}`),

  sources: () => request<{ count: number; sources: Source[] }>("/sources"),

  sourceDetail: (sourceId: string) =>
    request<{
      source: Source
      item_count: number
      items: Array<{ item_id: string; content: string; location: unknown }>
    }>(`/source/${sourceId}`),

  sourceFileUrl: (sourceId: string) => `${API_BASE_URL}/source/${sourceId}/file`,

  ingest: async (files: File[]): Promise<IngestResponse> => {
    const form = new FormData()
    files.forEach((f) => form.append("files", f))
    return request<IngestResponse>("/ingest", { method: "POST", body: form })
  },

  query: async (req: QueryRequest, backend?: string): Promise<QueryResponse> => {
    return request<QueryResponse>(
      `/query${backend ? `?backend=${backend}` : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      },
    )
  },

  queryFile: async (
    file: File,
    opts: {
      instruction?: string
      query_modality?: string
      inference_mode?: string
      top_k?: number
      rerank_top_k?: number
      backend?: string
    },
  ): Promise<QueryResponse> => {
    const form = new FormData()
    form.append("file", file)
    if (opts.instruction) form.append("instruction", opts.instruction)
    if (opts.query_modality) form.append("query_modality", opts.query_modality)
    if (opts.inference_mode) form.append("inference_mode", opts.inference_mode)
    if (opts.top_k != null) form.append("top_k", String(opts.top_k))
    if (opts.rerank_top_k != null)
      form.append("rerank_top_k", String(opts.rerank_top_k))
    if (opts.backend) form.append("backend", opts.backend)
    return request<QueryResponse>("/query_file", { method: "POST", body: form })
  },
}
