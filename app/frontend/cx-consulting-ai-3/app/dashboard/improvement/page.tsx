"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Download, RefreshCw } from "lucide-react"

interface Interaction {
  id: string
  timestamp: string
  question: string
  response: string
  context: string
  contextLength: number
  responseLength: number
}

export default function ModelImprovementPage() {
  const [interactions, setInteractions] = useState<Interaction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedInteraction, setSelectedInteraction] = useState<Interaction | null>(null)
  const pageSize = 10

  const fetchInteractions = async () => {
    setLoading(true)
    setError("")

    try {
      // This endpoint would need to be implemented in the backend
      const response = await fetch('/api/improvement/interactions')

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()

      // Format the interactions
      const formattedInteractions: Interaction[] = data.interactions.map((item: any) => ({
        id: item.id,
        timestamp: new Date(item.timestamp * 1000).toLocaleString(),
        question: item.question,
        response: item.response,
        context: item.context,
        contextLength: item.metadata.context_length,
        responseLength: item.metadata.response_length
      }))

      setInteractions(formattedInteractions)
    } catch (error) {
      console.error('Error fetching interactions:', error)
      setError('Failed to load interactions. This feature may not be fully implemented yet.')

      // For development, create some mock data
      const mockInteractions: Interaction[] = Array.from({ length: 15 }, (_, i) => ({
        id: `interaction-${i + 1}`,
        timestamp: new Date().toLocaleString(),
        question: `Sample customer question ${i + 1}?`,
        response: `This is a sample response to the customer's question. It would contain helpful information about CX consulting.`,
        context: `Sample context with information about CX methodologies and frameworks.`,
        contextLength: 500,
        responseLength: 200
      }))

      setInteractions(mockInteractions)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchInteractions()
  }, [])

  const totalPages = Math.ceil(interactions.length / pageSize)
  const paginatedInteractions = interactions.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  )

  const downloadInteraction = (interaction: Interaction) => {
    try {
      const data = {
        id: interaction.id,
        timestamp: interaction.timestamp,
        question: interaction.question,
        response: interaction.response,
        context: interaction.context,
        metadata: {
          contextLength: interaction.contextLength,
          responseLength: interaction.responseLength
        }
      }

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${interaction.id}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading interaction:', error)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Model Improvement</h1>
        <Button onClick={fetchInteractions} variant="outline" size="sm">
          <RefreshCw size={16} className="mr-2" />
          Refresh
        </Button>
      </div>

      <p className="text-muted-foreground">
        This page shows saved chat interactions that can be used for model training and improvement.
        Each interaction includes the user's question, AI response, and the context used for generation.
      </p>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded-md">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-muted rounded-lg overflow-hidden border">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Question</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Response Length</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-card divide-y divide-border">
                {paginatedInteractions.map((interaction) => (
                  <tr key={interaction.id} className="hover:bg-muted/50 cursor-pointer" onClick={() => setSelectedInteraction(interaction)}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{interaction.timestamp}</td>
                    <td className="px-6 py-4 text-sm">
                      <div className="truncate max-w-[300px]">{interaction.question}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">{interaction.responseLength} chars</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadInteraction(interaction);
                        }}
                      >
                        <Download size={16} />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Page {currentPage} of {totalPages}
              </div>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft size={16} />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight size={16} />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {selectedInteraction && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-card rounded-lg shadow-lg max-w-4xl w-full max-h-[80vh] overflow-auto p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">Interaction Details</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedInteraction(null)}
              >
                ✕
              </Button>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Timestamp</h3>
                <p>{selectedInteraction.timestamp}</p>
              </div>

              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Question</h3>
                <div className="p-3 bg-muted rounded-md">{selectedInteraction.question}</div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Response</h3>
                <div className="p-3 bg-muted rounded-md whitespace-pre-wrap">{selectedInteraction.response}</div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Context Used</h3>
                <div className="p-3 bg-muted rounded-md max-h-[200px] overflow-auto text-sm">{selectedInteraction.context}</div>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => downloadInteraction(selectedInteraction)}>
                  <Download size={16} className="mr-2" />
                  Download JSON
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
