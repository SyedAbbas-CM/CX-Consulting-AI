"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { SendIcon, PlusCircle, Loader2, BrainCircuit, ThumbsUp, FileJson2 } from "lucide-react"
import { useChatStore } from "../store/chatStore"
import { getChatHistory, markInteractionForRefinement } from "../lib/apiClient"
import { Message } from "../types/chat"
import type { QuestionRequest, SearchResult, DocumentGenerationConfig } from "@/src/types/api"
import * as apiClient from "../lib/apiClient"
import * as Dialog from '@radix-ui/react-dialog'
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import remarkGfm from 'remark-gfm'
import { DeliverableResponse, CXStrategyRequest, ROIAnalysisRequest, JourneyMapRequest } from "../types/deliverables"
import { useToast } from "./ui/use-toast"
import ReactMarkdownOrig from 'react-markdown'
import { ActionMenu } from './ActionMenu'
import { DeliverableGenerator } from './DeliverableGenerator'
const ReactMarkdown = ReactMarkdownOrig as unknown as React.FC<{
  children: React.ReactNode
  remarkPlugins?: any[]
}>

interface ChatMessage extends Message {
  sources?: SearchResult[];
}

interface ChatInterfaceProps {
  theme: "light" | "dark"
}

export default function ChatInterface({ theme }: ChatInterfaceProps) {
  const { currentChatId, setCurrentChatId } = useChatStore()
  const { currentProjectId } = useChatStore()

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const [showDeliverableModal, setShowDeliverableModal] = useState(false)
  const [selectedDeliverableTypeForModal, setSelectedDeliverableTypeForModal] = useState<string>("")

  const [refiningMessageIndex, setRefiningMessageIndex] = useState<number | null>(null)
  const [refinementSuccessIndex, setRefinementSuccessIndex] = useState<number | null>(null)
  const [refinementErrorIndex, setRefinementErrorIndex] = useState<number | null>(null)

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
        setMessages(history.map(m => ({ ...m, sources: [] } as ChatMessage)))
      } catch (error) {
        console.error("Failed to fetch chat history:", error)
        setMessages([{
          id: "error-hist",
          role: "assistant",
          content: `Error fetching chat history: ${error instanceof Error ? error.message : String(error)}`,
          timestamp: new Date().toISOString()
        }] as ChatMessage[])
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

      // Get active deliverable type from store
      const activeDeliverable = useChatStore.getState().activeDeliverableType;

      const optimisticUserMessage: ChatMessage = {
        id: `temp-user-${Date.now()}`,
        role: "user",
        content: userMessageContent,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, optimisticUserMessage])
      setIsLoading(true)

      try {
        const payload: QuestionRequest = {
          query: userMessageContent,
          conversation_id: currentChatId,
          project_id: currentProjectId,
          mode: activeDeliverable ? "document" : "chat", // Set mode based on active deliverable
          active_deliverable_type: activeDeliverable ?? undefined,
          // doc_config is not needed here if active_deliverable_type is set for the new flow
        };

        const data = await apiClient.askQuestion(payload);

        // If a deliverable was active, clear it from the store after successful send
        if (activeDeliverable) {
          useChatStore.getState().setActiveDeliverableType(null); // Auto-clear as per plan
        }

        const aiResponseMessage: ChatMessage = {
          id: data.conversation_id ? `ai-${data.conversation_id}-${Date.now()}` : `ai-resp-${Date.now()}`,
          role: "assistant",
          content: data.answer,
          timestamp: new Date().toISOString(),
          sources: data.sources || [],
        };
        setMessages((prev) => [...prev.filter(m => m.id !== optimisticUserMessage.id), optimisticUserMessage, aiResponseMessage]);
      } catch (error: any) {
        console.error("Error sending message:", error)
        const errorContent = error.errorData?.detail || error.message || "Failed to get a response."
        const errorResponse: ChatMessage = {
          id: `error-ai-${Date.now()}`,
          role: "assistant",
          content: `Sorry, I encountered an error: ${errorContent}`,
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev.filter(m => m.id !== optimisticUserMessage.id), errorResponse]);
      } finally {
        setIsLoading(false)
      }
    } else if (!currentChatId) {
      toast({title: "No Chat Selected", description: "Please select or create a chat first.", variant: "destructive"})
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

  const handleOpenDeliverableModal = (deliverableType: string) => {
    if (!currentProjectId) {
      toast({
        title: "Project Not Selected",
        description: "Please select a project to generate documents.",
        variant: "destructive",
      });
      return;
    }
    setSelectedDeliverableTypeForModal(deliverableType);
    setShowDeliverableModal(true);
  }

  const renderMessages = () => {
    if (isLoadingHistory) {
      return (
        <div className="flex justify-center items-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-gray-500 dark:text-gray-400" />
          <p className="ml-2 text-gray-500 dark:text-gray-400">Loading history...</p>
        </div>
      );
    }
    return messages.map((message, index) => (
      <div
        key={message.id || index}
        className={`flex flex-col max-w-[85%] sm:max-w-[75%] md:max-w-[70%] lg:max-w-[65%] ${
          message.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
        }`}
      >
        <div
          className={`p-3 rounded-xl shadow-md break-words text-sm ${
            message.role === "user"
              ? "bg-indigo-500 text-white dark:bg-indigo-600"
              : `bg-white text-gray-800 dark:bg-gray-800 dark:text-gray-100 border ${theme === 'dark' ? 'dark:border-gray-700' : 'border-gray-200'}`
          }`}
        >
          <div className="prose dark:prose-invert prose-sm max-w-none leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
          {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
              <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 flex items-center">
                <FileJson2 className="h-3.5 w-3.5 mr-1.5 opacity-80" /> Sources:
              </h4>
              <ul className="list-none space-y-1">
                {message.sources.map((source, idx) => (
                  <li
                    key={idx}
                    className="text-xs p-1.5 rounded-md bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors duration-150 ease-in-out"
                    title={source.text_snippet}
                  >
                    <div className="flex items-center justify-between">
                      <span className="truncate font-medium text-gray-600 dark:text-gray-300">
                        {source.source}
                      </span>
                      <span className="ml-2 px-1.5 py-0.5 text-[10px] rounded-full bg-gray-200 dark:bg-gray-500/80 text-gray-600 dark:text-gray-300">
                        {source.score.toFixed(2)}
                      </span>
                    </div>
                    {source.text_snippet && (
                       <p className="mt-0.5 text-gray-500 dark:text-gray-400 text-[11px] leading-tight truncate">
                         {source.text_snippet}
                       </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="flex items-center mt-1 px-1">
            <span className="text-xs text-gray-400 dark:text-gray-500">
                {message.role === "user" ? "You" : "Assistant"} - {formatTimestamp(message.timestamp ?? new Date().toISOString())}
            </span>
            {message.role === 'assistant' && (
              <Button
                variant="ghost"
                size="icon"
                title="Mark for refinement"
                onClick={() => handleRefineClick(index)}
                className="ml-1.5 p-0.5 h-auto w-auto disabled:opacity-50"
                disabled={refiningMessageIndex === index || refinementSuccessIndex === index}
              >
                {refiningMessageIndex === index ? <Loader2 className="h-3 w-3 animate-spin"/> : <ThumbsUp className={`h-3 w-3 ${refinementSuccessIndex === index ? 'text-green-500' : refinementErrorIndex === index ? 'text-red-500' : 'text-gray-400 hover:text-blue-500 dark:text-gray-500 dark:hover:text-blue-400'}`} />}
              </Button>
            )}
        </div>
      </div>
    ));
  };

  return (
    <div className={`flex flex-col h-full ${theme === "dark" ? "bg-gray-900" : "bg-gray-50"} focus:outline-none`} tabIndex={-1}>
      <div className="p-3 border-b flex justify-between items-center dark:border-gray-700">
          <div>
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                Chat: {currentChatId ? `${currentChatId.substring(0,8)}...` : "N/A"}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Project: {currentProjectId ? `${currentProjectId.substring(0,8)}...` : "N/A"}
              </p>
          </div>
          {currentProjectId && <ActionMenu currentProjectId={currentProjectId} onOpenDeliverableGenerator={handleOpenDeliverableModal} />}
      </div>

      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {renderMessages()}
        <div ref={messagesEndRef} />
      </div>

      <div className={`p-3 border-t ${theme === "dark" ? "border-gray-700" : "border-gray-200"} flex items-center space-x-2`}>
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={currentChatId ? "Type your message..." : "Select or create a chat to begin."}
          className={`flex-grow ${theme === "dark" ? "bg-gray-800 border-gray-700 text-white placeholder-gray-400" : "bg-white"}`}
          disabled={isLoading || isLoadingHistory || !currentChatId}
        />
        <Button
            onClick={handleSend}
            disabled={isLoading || isLoadingHistory || !input.trim() || !currentChatId }
            className={`px-4 py-2 text-white rounded-md ${theme === "dark" ? "bg-indigo-600 hover:bg-indigo-700" : "bg-indigo-500 hover:bg-indigo-600"}`}
        >
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendIcon className="h-4 w-4" />}
          <span className="sr-only">Send</span>
        </Button>
      </div>

      <Dialog.Root open={showDeliverableModal} onOpenChange={setShowDeliverableModal}>
        <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm data-[state=open]:animate-overlayShow" />
            <Dialog.Content
              className={`fixed top-1/2 left-1/2 w-[90vw] max-w-lg -translate-x-1/2 -translate-y-1/2
                         bg-white dark:bg-gray-800 p-6 rounded-lg shadow-xl focus:outline-none
                         data-[state=open]:animate-contentShow`}
            >
                <Dialog.Title className="text-xl font-semibold mb-1 text-gray-900 dark:text-gray-100">
                  Generate Document
                </Dialog.Title>
                <Dialog.Description className="text-sm text-gray-600 dark:text-gray-400 mb-5">
                  Select document type and provide parameters in JSON format.
                </Dialog.Description>

                <DeliverableGenerator />

                <Dialog.Close asChild className="mt-6 w-full">
                    <Button variant="outline">Close</Button>
                </Dialog.Close>
            </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
