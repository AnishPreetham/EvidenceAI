import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { Backend, QueryRequest } from "@/lib/types"

export function useHealth(backend: Backend) {
  return useQuery({
    queryKey: ["health", backend],
    queryFn: () => api.health(backend),
    refetchInterval: 15_000,
  })
}

export function useSources() {
  return useQuery({ queryKey: ["sources"], queryFn: api.sources })
}

export function useIngest() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (files: File[]) => api.ingest(files),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] })
      queryClient.invalidateQueries({ queryKey: ["health"] })
    },
  })
}

export function useRunQuery(backend: Backend) {
  return useMutation({
    mutationFn: (req: QueryRequest) => api.query(req, backend),
  })
}

export function useRunQueryFile(backend: Backend) {
  return useMutation({
    mutationFn: (args: {
      file: File
      instruction?: string
      query_modality?: string
      inference_mode?: string
      top_k?: number
      rerank_top_k?: number
    }) => api.queryFile(args.file, { ...args, backend }),
  })
}
