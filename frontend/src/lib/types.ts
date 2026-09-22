// Mirrors the real schemas.py exactly. Keep in sync with the locked backend
// contract — this file does not invent shapes, it reflects them.

export type Modality = "text" | "image" | "audio"

export type SourceType =
  | "pdf" | "doc" | "docx" | "txt" | "csv" | "image" | "audio" | "unknown"

export type RelationshipType =
  | "same_source" | "same_page" | "ocr_of" | "transcript_of"
  | "references" | "related_to"

export type InferenceMode = "local" | "local_vision"

export type QueryModality = "text" | "image" | "audio" | "document"

export type Backend = "offline" | "online" | "auto"

export type ErrorCode =
  | "OLLAMA_NOT_AVAILABLE" | "QWEN_MODEL_NOT_AVAILABLE" | "OCR_NOT_AVAILABLE"
  | "VISION_FAILED" | "TRANSCRIPTION_FAILED" | "UNSUPPORTED_FILE_TYPE"
  | "FILE_TOO_LARGE" | "FILE_NOT_FOUND" | "CORRUPT_FILE"
  | "INSUFFICIENT_EVIDENCE" | "INVALID_QUERY" | "SOURCE_NOT_FOUND"
  | "CONVERSION_UNAVAILABLE" | "INDEX_ERROR"

export interface SourceLocation {
  page_number: number | null
  section: string | null
  timestamp_start: number | null
  timestamp_end: number | null
  image_path: string | null
}

export interface Source {
  source_id: string
  filename: string
  source_type: SourceType
  file_path: string
  mime_type: string | null
  file_size: number | null
}

export interface ContentItem {
  item_id: string
  source_id: string
  modality: Modality
  content: string
  location: SourceLocation
  parent_id: string | null
  vector_id: number | null
  metadata: Record<string, unknown>
}

export interface Relationship {
  relationship_id: string
  source_item_id: string
  target_item_id: string
  relationship_type: RelationshipType
  confidence: number | null
  metadata: Record<string, unknown>
}

export interface RerankedResult {
  item: ContentItem
  retrieval_score: number
  rerank_score: number
  rank: number
}

export interface Citation {
  citation_id: number
  item_id: string
  source_id: string
  filename: string
  modality: Modality
  location: SourceLocation
  excerpt: string | null
}

export interface QueryRequest {
  query: string
  top_k: number
  rerank_top_k: number
  inference_mode: InferenceMode
  query_modality: QueryModality
  query_file_path?: string | null
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  retrieved_items: RerankedResult[]
  relationships: Relationship[]
  abstained: boolean
  confidence: number | null
  error_code: ErrorCode | null
  query_representation: string | null
  latency_ms: Record<string, number>
  technical_detail?: string | null
}

export interface IngestFileResult {
  filename: string
  source_id?: string
  source_type?: string
  items_indexed?: number
  relationships?: number
  success: boolean
  error?: string | null
  error_code?: ErrorCode | null
  warnings?: string[]
}

export interface IndexStats {
  sources: number
  items: number
  relationships: number
  vectors: number
  by_modality?: Record<string, number>
  index_path?: string
  db_path?: string
}

export interface IngestResponse {
  results: IngestFileResult[]
  indexed: number
  failed: number
  index: IndexStats
}

export interface InferenceStatus {
  backend: Backend
  label: string
  model: string
  ready: boolean
  message: string
  technical: string
  can_autostart?: boolean
  // AUTO mode only
  active_provider?: "online" | "offline" | null
  fallback?: boolean
  fallback_reason?: string | null
  online?: { ready: boolean; message: string; model: string; code: string | null }
  offline?: {
    ready: boolean
    message: string
    model: string
    code: string | null
    can_autostart?: boolean
  }
}

export interface HealthResponse {
  status: string
  backend?: Backend
  inference: InferenceStatus
  tesseract: { available: boolean; detail?: string }
  libreoffice: { available: boolean }
  models: Record<string, string>
  index: IndexStats
  detail?: string
}
