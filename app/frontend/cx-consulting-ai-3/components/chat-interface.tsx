"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { SendIcon, PlusCircle, Loader2, BrainCircuit, ThumbsUp } from "lucide-react"
import { useChatStore } from "../store/chatStore"
import { getChatHistory, markInteractionForRefinement } from "../lib/apiClient"
import { Message } from "../types/chat"
import * as apiClient from "../lib/apiClient"
import * as Dialog from '@radix-ui/react-dialog'
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import remarkGfm from 'remark-gfm'
import { DeliverableResponse, CXStrategyRequest, ROIAnalysisRequest, JourneyMapRequest } from "../types/deliverables"
import { useToast } from "./ui/use-toast"
import ReactMarkdownOrig from 'react-markdown'
const ReactMarkdown = ReactMarkdownOrig as unknown as React.FC<{
  children: React.ReactNode
  remarkPlugins?: any[]
}>

interface ChatInterfaceProps {
  theme: "light" | "dark"
}

export default function ChatInterface({ theme }: ChatInterfaceProps) {
  const { currentChatId, setCurrentChatId } = useChatStore()
  const { currentProjectId } = useChatStore()

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false)
  const [selectedTask, setSelectedTask] = useState<string | null>(null)
  const [taskFormData, setTaskFormData] = useState<any>({})
  const [isGeneratingTask, setIsGeneratingTask] = useState(false)

  const [refiningMessageIndex, setRefiningMessageIndex] = useState<number | null>(null)
  const [refinementSuccessIndex, setRefinementSuccessIndex] = useState<number | null>(null)
  const [refinementErrorIndex, setRefinementErrorIndex] = useState<number | null>(null)

  const [showDeliverableModal, setShowDeliverableModal] = useState(false)
  const [deliverableType, setDeliverableType] = useState<string>("proposal")
  const [generatingDeliverable, setGeneratingDeliverable] = useState(false)

  const { toast } = useToast()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    const fetchHistory = async () => {
      if (!currentChatId) {
        setMessages([])
        return
      }

      console.log(`ChatUI: Fetching history for chat ID: ${currentChatId}`)
      setIsLoadingHistory(true)
      setMessages([])
      try {
        const history = await getChatHistory(currentChatId)
        console.log(`ChatUI: Fetched ${history.length} messages.`)
        setMessages(history)
      } catch (error) {
        console.error("Failed to fetch chat history:", error)
        setMessages([{
          id: "error-hist",
          role: "assistant",
          content: `Error fetching chat history: ${error instanceof Error ? error.message : String(error)}`,
          timestamp: new Date().toISOString()
        }])
      } finally {
        setIsLoadingHistory(false)
      }
    }

    fetchHistory()
  }, [currentChatId])

  const handleSend = async () => {
    if (input.trim() && currentChatId) {
      const userMessageContent = input
      setInput("")

      const optimisticUserMessage: Message = {
        id: `temp-${Date.now()}`,
        role: "user",
        content: userMessageContent,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, optimisticUserMessage])
      setIsLoading(true)

      try {
        const token = localStorage.getItem("authToken")
        if (!token) throw new Error("User not authenticated")

        const response = await fetch("/api/ask", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            query: userMessageContent,
            conversation_id: currentChatId,
            project_id: currentProjectId
          }),
        })

        if (response.status === 401) throw new Error("Authentication failed")
        if (!response.ok) throw new Error(`API error: ${response.status}`)

        const data = await response.json()

        const aiResponse: Message = {
          id: `temp-ai-${Date.now()}`,
          role: "assistant",
          content: data.answer,
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, aiResponse])
      } catch (error) {
        console.error("Error sending message:", error)
        const errorResponse: Message = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: `Error: ${error instanceof Error ? error.message : "Failed to send message."}`,
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev.filter(m => m.id !== optimisticUserMessage.id), errorResponse])
      } finally {
        setIsLoading(false)
      }
    } else if (!currentChatId) {
      alert("Please select a chat first.")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const formatTimestamp = (isoString: string): string => {
    try {
      return new Date(isoString).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    } catch {
      return isoString
    }
  }

  const handleTaskFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setTaskFormData({
      ...taskFormData,
      [e.target.name]: e.target.value
    })
  }

  const handleTaskSelect = (task: string) => {
    setSelectedTask(task)
    setTaskFormData({})
  }

  const handleGenerateTask = async () => {
    if (!selectedTask || !currentChatId) return

    const currentProjectId = useChatStore.getState().currentProjectId

    setIsGeneratingTask(true)
    let taskFunction
    let requestData: any = {
      ...taskFormData,
      conversation_id: currentChatId,
      project_id: currentProjectId
    }

    try {
      let taskName = "Task"
      switch (selectedTask) {
        case 'proposal':
          taskFunction = apiClient.generateProposal
          taskName = "Proposal"
          if (!taskFormData.client_name || !taskFormData.industry || !taskFormData.challenges) {
            throw new Error("Client Name, Industry, and Challenges are required for Proposal.")
          }
          requestData = { ...requestData } as CXStrategyRequest
          break
        case 'roi':
          taskFunction = apiClient.generateRoiAnalysis
          taskName = "ROI Analysis"
          if (!taskFormData.client_name || !taskFormData.industry || !taskFormData.project_description || !taskFormData.current_metrics) {
            throw new Error("Client Name, Industry, Project Description, and Current Metrics are required for ROI Analysis.")
          }
          requestData = { ...requestData } as ROIAnalysisRequest
          break
        case 'journey_map':
          taskFunction = apiClient.generateJourneyMap
          taskName = "Journey Map"
          if (!taskFormData.client_name || !taskFormData.industry || !taskFormData.persona || !taskFormData.scenario) {
            throw new Error("Client Name, Industry, Persona, and Scenario are required for Journey Map.")
          }
          requestData = { ...requestData } as JourneyMapRequest
          break
        default:
          throw new Error("Invalid task selected")
      }

      const result = await taskFunction(requestData)

      const taskResultMessage: Message = {
        id: `task-${Date.now()}`,
        role: 'assistant',
        content: `**${taskName} Generation Complete:**\n\n${result.content}`,
        timestamp: new Date().toISOString()
      }
      setMessages((prev) => [...prev, taskResultMessage])
      setIsTaskModalOpen(false)
      setSelectedTask(null)
      setTaskFormData({})
    } catch (error) {
      console.error(`Error generating ${selectedTask}:`, error)
      alert(`Failed to generate ${selectedTask}: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsGeneratingTask(false)
    }
  }

  const handleRefineClick = async (messageIndex: number) => {
    if (!currentChatId || refiningMessageIndex === messageIndex) return

    setRefiningMessageIndex(messageIndex)
    setRefinementSuccessIndex(null)
    setRefinementErrorIndex(null)
    console.log(`ChatUI: Attempting to refine message index ${messageIndex} in chat ${currentChatId}`)

    try {
      const result = await markInteractionForRefinement(currentChatId, messageIndex)
      console.log("Refinement API success:", result)
      setRefinementSuccessIndex(messageIndex)
    } catch (err) {
      console.error("ChatUI: Error marking interaction for refinement:", err)
      setRefinementErrorIndex(messageIndex)
    } finally {
      setRefiningMessageIndex(null)
    }
  }

  const handleOpenDeliverableModal = () => {
    if (!currentChatId) {
      toast({ variant: "destructive", title: "Error", description: "Please select or create a chat first." })
      return
    }
    setShowDeliverableModal(true)
  }

  const handleGenerateDeliverable = async (formData: any) => {
    if (!currentChatId) return
    setGeneratingDeliverable(true)
    try {
      let response: DeliverableResponse | null = null
      const baseRequestData = {
        client_name: formData.client_name,
        industry: formData.industry,
        project_id: currentProjectId,
        conversation_id: currentChatId
      }

      if (deliverableType === 'proposal') {
        const requestData: CXStrategyRequest = {
          ...baseRequestData,
          challenges: formData.challenges
        }
        response = await apiClient.generateProposal(requestData)
      } else if (deliverableType === 'roi') {
        const requestData: ROIAnalysisRequest = {
          ...baseRequestData,
          project_description: formData.project_description,
          current_metrics: formData.current_metrics
        }
        response = await apiClient.generateRoiAnalysis(requestData)
      } else if (deliverableType === 'journey') {
        const requestData: JourneyMapRequest = {
          ...baseRequestData,
          persona: formData.persona,
          scenario: formData.scenario
        }
        response = await apiClient.generateJourneyMap(requestData)
      }

      if (response) {
        console.log("Generated Deliverable:", response)
        const deliverableMessage: Message = {
          id: `deliverable-${Date.now()}`,
          role: 'assistant',
          content: `**${deliverableType.charAt(0).toUpperCase() + deliverableType.slice(1)} Deliverable Generated:**\n\n${response.content}`,
          timestamp: new Date().toISOString()
        }
        setMessages((prev) => [...prev, deliverableMessage])
        toast({ title: "Success", description: `Successfully generated ${deliverableType} deliverable.` })
      } else {
        throw new Error("No response received from deliverable generation.")
      }
    } catch (error: any) {
      console.error("Error generating deliverable:", error)
      toast({ variant: "destructive", title: "Generation Failed", description: `Error generating ${deliverableType}: ${error.message}` })
    } finally {
      setGeneratingDeliverable(false)
      setShowDeliverableModal(false)
    }
  }

  return (
    <div
      className={`flex flex-col border rounded-lg overflow-hidden ${
        theme === "dark" ? "border-gray-800 bg-[#0f1117]" : "border-gray-200 bg-gray-50"
      }`}
      style={{ height: "100%" }}
    >
      <div className="p-3 border-b flex justify-between items-center">
        <h3 className="font-medium">Chat {currentChatId ? `(${currentChatId.substring(0, 8)}...)` : ''}</h3>
        <Dialog.Root open={isTaskModalOpen} onOpenChange={setIsTaskModalOpen}>
          <Dialog.Trigger asChild>
            <Button variant="ghost" size="sm" disabled={!currentChatId} title="Generate Deliverable">
              <BrainCircuit size={18} className="mr-1" /> Tasks
            </Button>
          </Dialog.Trigger>
          <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 bg-black/50 data-[state=open]:animate-overlayShow" />
            <Dialog.Content
              className={`fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 p-6 rounded-lg shadow-lg w-[90vw] max-w-lg max-h-[85vh] overflow-y-auto focus:outline-none data-[state=open]:animate-contentShow ${
                theme === 'dark' ? 'bg-gray-900 border border-gray-700' : 'bg-white'
              }`}
            >
              <Dialog.Title className="text-lg font-medium mb-4">Generate Deliverable</Dialog.Title>

              {!selectedTask ? (
                <div className="flex flex-col space-y-2">
                  <Button onClick={() => handleTaskSelect('proposal')}>Proposal</Button>
                  <Button onClick={() => handleTaskSelect('roi')}>ROI Analysis</Button>
                  <Button onClick={() => handleTaskSelect('journey_map')}>Journey Map</Button>
                </div>
              ) : (
                <div>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedTask(null)} className="mb-2">&larr; Back to Tasks</Button>
                  <h4 className="font-semibold mb-3 capitalize">{selectedTask.replace('_', ' ')}</h4>
                  <form onSubmit={(e) => { e.preventDefault(); handleGenerateTask(); }} className="space-y-3">
                    <div><Label htmlFor="client_name">Client Name</Label><Input name="client_name" id="client_name" value={taskFormData.client_name || ''} onChange={handleTaskFormChange} required /></div>
                    <div><Label htmlFor="industry">Industry</Label><Input name="industry" id="industry" value={taskFormData.industry || ''} onChange={handleTaskFormChange} required /></div>

                    {selectedTask === 'proposal' && (
                      <div><Label htmlFor="challenges">Challenges</Label><Textarea name="challenges" id="challenges" value={taskFormData.challenges || ''} onChange={handleTaskFormChange} required /></div>
                    )}
                    {selectedTask === 'roi' && (
                      <>
                        <div><Label htmlFor="project_description">Project Description</Label><Textarea name="project_description" id="project_description" value={taskFormData.project_description || ''} onChange={handleTaskFormChange} required /></div>
                        <div><Label htmlFor="current_metrics">Current Metrics</Label><Textarea name="current_metrics" id="current_metrics" value={taskFormData.current_metrics || ''} onChange={handleTaskFormChange} required /></div>
                      </>
                    )}
                    {selectedTask === 'journey_map' && (
                      <>
                        <div><Label htmlFor="persona">Persona</Label><Textarea name="persona" id="persona" value={taskFormData.persona || ''} onChange={handleTaskFormChange} required /></div>
                        <div><Label htmlFor="scenario">Scenario</Label><Textarea name="scenario" id="scenario" value={taskFormData.scenario || ''} onChange={handleTaskFormChange} required /></div>
                      </>
                    )}

                    <div className="flex justify-end mt-4">
                      <Button type="submit" disabled={isGeneratingTask}>
                        {isGeneratingTask ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                        Generate {selectedTask.replace('_', ' ')}
                      </Button>
                    </div>
                  </form>
                </div>
              )}

              <Dialog.Close asChild>
                <button className="absolute top-3 right-3 inline-flex h-6 w-6 appearance-none items-center justify-center rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400" aria-label="Close">
                  &times;
                </button>
              </Dialog.Close>
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {isLoadingHistory ? (
          <div className="flex justify-center items-center h-full">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        ) : messages.length === 0 && currentChatId ? (
          <div className="text-center text-gray-500">(Chat history is empty)</div>
        ) : messages.length === 0 && !currentChatId ? (
          <div className="text-center text-gray-500">(Select a chat from the sidebar)</div>
        ) : (
          messages.map((message, index) => (
            <div
              key={message.id}
              className={`flex ${message.role === "assistant" ? "items-start" : "items-start justify-end"}`}
            >
              {message.role === "assistant" && (
                <div
                  className={`flex items-center justify-center w-8 h-8 rounded-full mr-2 flex-shrink-0 ${
                    theme === "dark"
                      ? "bg-blue-600"
                      : "bg-blue-500"
                  }`}
                >
                  <span className="text-white text-xs font-bold">AI</span>
                </div>
              )}
              <div className="max-w-[80%]">
                <div
                  className={`p-4 rounded-lg ${
                    message.role === "assistant"
                      ? theme === "dark"
                        ? "bg-gray-800"
                        : "bg-white border border-gray-200"
                      : theme === "dark"
                        ? "bg-blue-600"
                        : "bg-blue-500 text-white"
                  }`}
                >
                  <div className="prose dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                </div>
                <p className={`text-xs mt-1 ${theme === "dark" ? "text-gray-500" : "text-gray-500"} ${message.role === 'user' ? 'text-right' : ''}`}>
                {formatTimestamp(message.timestamp ?? new Date().toISOString())}
                </p>
                {message.role === 'assistant' && (
                  <div className="flex justify-end mt-1 opacity-50 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className={`h-6 w-6 p-0 
                        ${refinementSuccessIndex === index ? 'text-green-500' : 'text-gray-400 hover:text-blue-500'}
                        ${refinementErrorIndex === index ? 'text-red-500' : ''}
                      `}
                      onClick={() => handleRefineClick(index)}
                      disabled={refiningMessageIndex === index}
                      title="Mark this response for refinement"
                    >
                      {refiningMessageIndex === index ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <ThumbsUp className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className={`p-3 border-t ${theme === "dark" ? "border-gray-800" : "border-gray-200"}`}>
        <div className="flex items-center space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={currentChatId ? "Type your message..." : "Select a chat first..."}
            className={theme === "dark" ? "bg-gray-800 border-gray-700" : "bg-white"}
            disabled={isLoading || isLoadingHistory || !currentChatId}
          />
          <Button
            onClick={handleSend}
            size="icon"
            className={theme === "dark" ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600"}
            disabled={isLoading || isLoadingHistory || !input.trim() || !currentChatId}
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendIcon size={18} />}
          </Button>
        </div>
      </div>
    </div>
  )
}
